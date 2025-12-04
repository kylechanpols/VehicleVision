import clip
from .constants import Constants
import torch
from torchvision.transforms.transforms import Compose
import clip
from PIL import Image
from typing import List
from pathlib import Path

class CLIPImageProcessor:
    clip_model_name: str = "ViT-B/32"
    device: str
    model: clip.model.CLIP
    preprocess: Compose
    class_prompts: List[str]
    condition_prompts: List[str]

    def __init__(self, clip_model_name:str | None = None, class_prompts: List[str] = Constants.CLASS_PROMPTS, condition_prompts: List[str] = Constants.CONDITION_PROMPTS):
        self.clip_model_name = clip_model_name if clip_model_name else "ViT-B/32"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(self.clip_model_name, device=self.device)
        self.class_prompts = class_prompts
        self.condition_prompts = condition_prompts

    def process_image(self, image: Path):
        raw_img = Image.open(image)
        image = self.preprocess(raw_img).unsqueeze(0).to(self.device)
        text, condition = clip.tokenize(self.class_prompts).to(self.device), clip.tokenize(self.condition_prompts).to(self.device) # Challenge of Using CLIP - if designing this as an Agent

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            text_features_condition = model.encode_text(condition)
            
            logits_per_image, logits_per_text = self.model(image, text)
            logits_per_image_cond, logits_per_text_cond = self.model(image, condition)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            probs_cond = logits_per_image_cond.softmax(dim=-1).cpu().numpy()

        return probs, probs_cond
    
    def process_image_batch(self, img_paths: List[Path]):
            
        # Load and preprocess each image in the batch
        batch_images = []
        for img_path in img_paths:
            raw_img = Image.open(img_path)
            # preprocess returns a tensor, so we collect them
            preprocessed_img = self.preprocess(raw_img)
            batch_images.append(preprocessed_img)

        # Stack the preprocessed images into a batch tensor
        image_batch = torch.stack(batch_images).to(self.device)
        
        # Tokenize text prompts (same for all images in batch)
        text = clip.tokenize(self.class_prompts).to(self.device)
        condition = clip.tokenize(self.condition_prompts).to(self.device)
        
        with torch.no_grad():
            # Encode the batch of images
            image_features = self.model.encode_image(image_batch)
            text_features = self.model.encode_text(text)
            text_features_condition = self.model.encode_text(condition)
            
            # Get logits for the batch
            logits_per_image, logits_per_text = self.model(image_batch, text)
            logits_per_image_cond, logits_per_text_cond = self.model(image_batch, condition)
            
            # Get probabilities for each image in the batch
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            probs_cond = logits_per_image_cond.softmax(dim=-1).cpu().numpy()
            
        return probs, probs_cond # [N_Prompts * batch_size]