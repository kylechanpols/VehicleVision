from typing import Annotated, Literal
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
import pandas as pd
import sys
from pathlib import Path

# TODO: Maybe we can do this another way?
sys.path.append("../../")

from VehicleVision.api.serving import load_model

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

class PredictionParams(BaseModel):
    suv_prob : float = Field(0.0)
    vert_prob: float = Field(0.0)
    sedan_prob : float = Field(0.0)
    coupe_prob : float = Field(0.0)
    hatchback_prob: float = Field(0.0)
    window_damage_prob: float = Field(0.0)
    door_damage_prob: float = Field(0.0)
    bumper_damage_prob: float = Field(0.0)
    hood_damage_prob:float = Field(0.0)
    excellent_cond_prob:float = Field(0.0)
    average_cond_prob: float = Field(0.0)
    year : int = Field(2000)
    make : str = Field("")
    model_name: str = Field("")
    mileage: int = Field(100000)
    title_status: Literal["Clean", "Salvage", "Unknown"] = "Clean"
    transmission: Literal["Automatic", "Manual", "Sequential", "Unknown"] = "Automatic"
    model_path: str = Field("../models/model")

# for now we default to clean examples predictions
@app.get("/predict/")
async def predict(query: Annotated[PredictionParams, Query()]):
    """Predict vehicle price based on vehicle characteristics and condition.
    
    This endpoint loads a pre-trained machine learning model and generates a price
    prediction for a vehicle based on its type, damage assessment, condition, and
    basic specifications.
    
    Args:
        query: PredictionParams
            suv_prob: float
                The predicted probability of this vehicle being an SUV. This is the maximum predicted probability from CLIP when processing all photos in a vehicle listing.
            vert_prob: float
                The predicted probability of this vehicle being a convertible. This is the maximum predicted probability from CLIP when processing all photos in a vehicle listing.
            sedan_prob: float
                The predicted probability of this vehicle being a 4-door sedan. This is the maximum predicted probability from CLIP when processing all photos in a vehicle listing.
            coupe_prob: float
                The predicted probability of this vehicle being a 2-door coupe. This is the maximum predicted probability from CLIP when processing all photos in a vehicle listing.
            hatchback_prob: float
                The predicted probability of this vehicle being a 3-door or 5-door hatchback. This is the maximum predicted probability from CLIP when processing all photos in a vehicle listing.
            window_damage_prob: float
                The predicted probability of this vehicle having window damage. This is the maximum predicted probability from CLIP when processing all photos in a vehicle listing.
            door_damage_prob: float
                The predicted probability of this vehicle having a damaged door. This is the maximum predicted probability from CLIP when processing all photos in a vehicle listing.
            bumper_damage_prob: float
                The predicted probability of this vehicle having a damaged bumper. This is the maximum predicted probability from CLIP when processing all photos in a vehicle listing.
            hood_damage_prob: float
                The predicted probability of this vehicle having a damaged hood. This is the maximum predicted probability from CLIP when processing all photos in a vehicle listing.
            excellent_cond_prob: float
                The predicted probability of this vehicle is in excellent condition. This is the maximum predicted probability from CLIP when processing all photos in a vehicle listing.
            average_cond_prob: float
                The predicted probability of this vehicle is in average condition. This is the maximum predicted probability from CLIP when processing all photos in a vehicle listing.
            year: int
                Year of vehicle
            make: str
                Vehicle make
            model_name: str
                Vehicle Model Name
            mileage: int
                Vehicle mileage, in US miles
            title_status: str
                Vehicle title status, must be one of ["Clean", "Salvage", "Unknown"].
            transmission: str
                Vehicle transmission, must be one of ["Automatic", "Manual", "Sequential" ,"Unknown"]. Automatic transmission includes all variations of the automatic gearbox (e.g. Automatic Transmission, CVT, DCT).
            model_path: str
                Location of the trained machine learning model to use
    
    Returns:
        JSON response containing the predicted price with key "pred" (float).
    
    Raises:
        RequestValidationError: If required parameters (make, model_name) are missing
            or if parameter types don't match expected formats.
        FileNotFoundError: If the model file at model_path doesn't exist.
    
    Example:
        GET /predict/?make=Honda&model_name=Civic&year=2023&mileage=50000
        
        Response:
        {
            "pred": 18500.75
        }
    """
    model = load_model(query.model_path)
    data = pd.DataFrame(query.model_dump(), index=[0])
    return {"pred": float(model.predict(data).squeeze())}