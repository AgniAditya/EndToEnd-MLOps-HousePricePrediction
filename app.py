from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import os

# Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices using machine learning",
    version="1.0.0"
)

# Define input data model
class HouseFeatures(BaseModel):
    Title: int
    Bathroom: float
    Carpet_Area: float
    location: int
    Transaction: int
    Furnishing: int
    Balcony: float
    facing: int
    Price_in_rupees: float
    Status: int
    Society: int
    Floor: int

    class Config:
        schema_extra = {
            "example": {
                "Title": 1,
                "Bathroom": 2.0,
                "Carpet Area": 1000.0,
                "location": 1,
                "Transaction": 1,
                "Furnishing": 1,
                "Balcony": 1.0,
                "facing": 1,
                "Price (in rupees)": 5000.0,
                "Status": 1,
                "Society": 1,
                "Floor": 1
            }
        }

# Load model and scaler from local files
try:
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Load the model and scaler
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    logging.info("Model and scaler loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise

@app.post("/predict")
async def predict_price(features: HouseFeatures):
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([features.dict()])
        
        # Scale features
        scaled_features = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        
        return {
            "predicted_price": float(prediction[0]),
            "prediction_confidence": "high"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)