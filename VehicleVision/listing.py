import os
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import warnings
import craigslistscraper as cs

from VehicleVision.download import download_images
from VehicleVision.image.image import CLIPImageProcessor
from VehicleVision.image.constants import Constants as ImageProcessorConstants
from VehicleVision.llm.model_inference import LLMDescriptionParser

class ListingMissingFieldError(Exception):
    pass

class Listing:

    uid: str
    price: int
    url: str
    title: str
    description: str
    image_urls: List[str] | None = None
    image_locs: List[str] | None = None
    clip_class_prompts: List[str] | None = None
    clip_probs: np.ndarray | None = None
    clip_condition_prompts: List[str] | None = None
    clip_cond_probs: np.ndarray | None = None
    text_features: Dict[str, List[Any]] | None = None
    llm_parser: LLMDescriptionParser | None = None # invoke the LLM only once to prevent unnecessarily re-initializing the chat model

    def __init__(self, **kwargs):
        self.uid = kwargs.get("uid")
        self.price = kwargs.get("price")
        self.url = kwargs.get("url")
        self.title = kwargs.get("title")
        self.description = kwargs.get("description")
        self.image_locs = kwargs.get("image_locs")

    def __call__(self):
        print(f"A Vehicle Listing object. UID: {self.uid} Titled: {self.title}")

    @classmethod
    def from_cs(cls,
            cs_ad: cs.ad.Ad,
            images_main_path: Path| None = None,
            download_images_bool: bool = True):
        if not cs_ad.d_pid:
            raise ListingMissingFieldError("uid")
        if not cs_ad.price:
            raise ListingMissingFieldError("price")
        if not cs_ad.url:
            raise ListingMissingFieldError("url")
        if not cs_ad.title:
            raise ListingMissingFieldError("title")

        if download_images_bool:
            parent_folder = images_main_path if images_main_path else Path(os.getcwd())
            download_loc = parent_folder / "img" / str(cs_ad.d_pid)
            statuses = download_images(cs_ad.image_urls, download_loc, large=True)
            image_locs = [img['filepath'] for img in statuses['successful']]
        return Listing(
            uid=cs_ad.d_pid,
            title=cs_ad.title,
            price=cs_ad.price,
            url=cs_ad.url,
            description = cs_ad.description,
            image_urls = cs_ad.image_urls,
            image_locs = image_locs,
        )
    
    def process_images(self):
        image_processor = CLIPImageProcessor()
        self.clip_probs, self.clip_cond_probs = image_processor.process_image_batch(self.image_locs)
        self.clip_class_prompts, self.clip_condition_prompts = ImageProcessorConstants.CLASS_PROMPTS, ImageProcessorConstants.CONDITION_PROMPTS
    
    def parse_description(self):
        if not self.llm_parser:
            llm_parser = LLMDescriptionParser()
        self.text_features_raw = llm_parser.invoke(self.description).content
        try:
            self.text_features = json.loads(self.text_features_raw)
        except json.JSONDecodeError:
            warnings.warn(f"Couldn't parse the description for {self.uid}, skipped.")

    def to_dict(self):
        return {
            "uid": self.uid,
            "price": self.price,
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "image_urls": self.image_urls,
            "image_locs": self.image_locs,
            "clip_class_prompts": self.clip_class_prompts,
            "clip_probs": self.clip_probs,
            "clip_condition_prompts": self.clip_condition_prompts,
            "clip_cond_probs": self.clip_cond_probs,
            "text_features": self.text_features,
        }
    
    def to_json(self):
        return json.dumps(self.to_dict())
        

