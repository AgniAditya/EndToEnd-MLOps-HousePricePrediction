from fastapi import FastAPI, HTTPException
from steps.ingest_data import ingestdata
import joblib
from sklearn.preprocessing import LabelEncoder
import logging
from pydantic import BaseModel, Field
import json
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    model = joblib.load('./models/model.pkl')
    with open('./models/label_encoders.json', 'r') as f:
        encoders_data = json.load(f)
        label_encoders = {}
        for col, classes in encoders_data.items():
            encoder = LabelEncoder()
            encoder.classes_ = np.array(classes)
            label_encoders[col] = encoder
    logger.info("Model has Started")
except Exception as e:
    logger.error(f"Error loading model or encoders: {str(e)}")
    raise

app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices based on various features",
    version="1.0.0"
)

class InputFeatures(BaseModel):
    Title: str = Field(..., description="Title of the property")
    Bathroom: int = Field(..., ge=0, description="Number of bathrooms")
    Carpet_Area: int = Field(..., ge=0, description="Carpet area in square feet")
    location: str = Field(..., description="Location of the property")
    Transaction: str = Field(..., description="Type of transaction")
    Furnishing: str = Field(..., description="Furnishing status")
    Balcony: int = Field(..., ge=0, description="Number of balconies")
    facing: str = Field(..., description="Direction the property faces")
    Price_in_rupees: int = Field(..., ge=0, description="Price in rupees")
    Status: str = Field(..., description="Status of the property")
    Society: str = Field(..., description="Society name")
    Floor: str = Field(..., description="Floor information")

class PredictionResponse(BaseModel):
    predicted_price: float
    confidence: float = Field(..., ge=0, le=1)
    status: str = "success"

@app.get('/')
def main():
    return {
        "message": "House Price Prediction System API",
        "status": "operational",
        "version": "1.0.0"
    }

@app.post('/predict', response_model=PredictionResponse)
async def predict(data: InputFeatures):
    try:
        input_dict = data.dict()
        df = ingestdata(input_dict)

        for col, encoder in label_encoders.items():
            if col in df.columns:
                if df[col].iloc[0] not in encoder.classes_:
                    df[col] = encoder.transform([encoder.classes_[0]])
                else:
                    df[col] = encoder.transform(df[col])

        expected_columns = model.feature_names_in_
        df = df[expected_columns]

        prediction = model.predict(df)
        
        confidence = 0.95
        
        return PredictionResponse(
            predicted_price=float(prediction[0]),
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

@app.get('/health')
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }