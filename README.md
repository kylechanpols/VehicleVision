# Vehicle Vision: AI powered used vehicle price estimator

Vehicle Vision is a project that aims to predict a used vehicle listing price through textual and visual information about a vehicle. The project trains a machine learning model using the generated features from a vehicle's photos and listing text.


## Image Processing
Vehicle images are parsed by the pre-trained image-to-text model CLIP. Each image is evaluated against a collection of textual prompts such as "Window Damage", "Door Damage", etc. 

## Text Parsing
The text from the listing is parsed by queries through an Llama model, which then extracts the relevant information from the listing such as the year of vehicle, vehicle make, model name, mileage information & etc. and structure the data for further processing.

After training, the model can predict the selling price for a vehicle not seen by the model given its photos and textual description of its condition:

# Model Training

<img width="915" height="521" alt="Screenshot 2025-12-03 at 3 41 29 PM" src="https://github.com/user-attachments/assets/46a17cd0-43f0-4ef8-8c51-45de753fdb5c" />

# Model Inference

<img width="837" height="493" alt="Screenshot 2025-12-03 at 3 41 41 PM" src="https://github.com/user-attachments/assets/85ed8a38-a85f-4921-a3d1-337e879cdf8f" />

# Model Performance

A quick example shown in the sandbox folder shows that a simple implementation of this approach was able to predict the used vehicle prices for a small sample of a mix of commonly found sedans and SUVs in the San Francisco Bay Area in August 2025 with a Root Mean Squared Error (RMSE) of $1500.

For the modeling process and outcome, please consult `sandbox/modeling.ipynb`.

# API 

An API is available for making model inference with unseen vehicle photos and listing text. The API implementation can be found in `VehicleVision/VehicleVision/api`. The API is implemented using FastAPI v0.123.5. 
The API must be hosted by using `fastapi run app.py`.
After processing the photos and listing text, pass the processed data as a URL query for the `predict` method inside the app. For example: `http://www.vehiclevision.com/predict?year=2020&make=toyota&model_name=corolla&mileage=120000&title_status=Clean&transmission=Automatic&`. The API then returns a JSON response of the predicted selling price of the vehicle.
