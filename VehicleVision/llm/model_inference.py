from typing import Any
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from pydantic import BaseModel, Field
DEFAULT_MODEL_NAME = "llama3.1:8b"

class ParsedDescription(BaseModel):
    """A parsed vehicle listing description"""
    year: str = Field(description="Year of vehicle")
    make: str = Field(description="Make of vehicle")
    model_name: str = Field(description="Model name of vehicle")
    trim: str = Field(description="Vehicle trim (if available)")
    mileage: int = Field(description="Mileage (in miles)")
    title_status: str = Field(description="clean or branded/ salvage (if available)")
    transmission: str = Field(description="Automatic or Manual (if available)")
    engine_cylinders: int = Field(description="Engine Size, in the number of cylinders (if available)")
    displacement: float = Field(description="Engine Displacement, in Litres (if available)")

class LLMDescriptionParser:
    model_name: str
    chat_model: ChatOllama # TODO: Change this when deploying to external service
    sys_prompt: str = """
    You are an expert used car salesperson. In a minute you will receive a description for a vehicle listing. Please
    extract the following information from the description:
    - Year of vehicle
    - Make of vehicle
    - Model name of vehicle
    - Vehicle trim (if available)
    - Mileage (in miles)
    - Title status: clean or branded/ salvage (if available)
    - Transmission type: Automatic or Manual (if available)
    - Engine Size, in the number of cylinders (if available)
    - Engine Displacement, in Litres (if available)
    
    If you are not sure if the information is available or not, just return an empty field.

    Return only the extracted information, do not return any other output.\n{format_instructions}\n
    """

    def __init__(self, model_name:str | None = None):
        model_name = model_name if model_name else DEFAULT_MODEL_NAME
        self.chat_model = ChatOllama(model =model_name)

    def get_sys_template(self, sys_prompt: str | None = None):
        return sys_prompt if sys_prompt else self.sys_prompt

    def get_parser(self, parser: Any| None = None):
        return parser if parser else PydanticOutputParser(pydantic_object=ParsedDescription)
    
    def invoke(self, query:str, smoke_test:bool=False):
        parser = self.get_parser()
        sys_template = self.get_sys_template()
        system_message_prompt = SystemMessagePromptTemplate.from_template(sys_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template("{query}")
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        messages = chat_prompt.format_prompt(
            format_instructions=parser.get_format_instructions(),
            query=query,
        ).to_messages()
        if smoke_test:
            return messages
        else:
            return self.chat_model.invoke(messages)

# TODO: each brand probably needs its own context - the AI doesn't have any right now
# psuedo code:
# invoke 1
# check make
# invoke 2
# get context about vehicle make
# add context to enrich the needed details, e.g. model name, trim, transmission ,engine type, etc.

