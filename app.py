# FastAPI application for House Price Prediction API
# This file serves as the main entry point for the prediction service
# It provides REST API endpoints for making house price predictions
from fastapi import FastAPI, HTTPException
from steps.ingest_data import ingestdata
import joblib
from sklearn.preprocessing import LabelEncoder
import logging
from pydantic import BaseModel, Field
import json
import numpy as np

# Configure logging for the application
# This sets up logging to track application events and errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model and label encoders at startup
# This ensures the model is ready for predictions when the API starts
try:
    # Load the trained Random Forest model from the models directory
    # This model was saved during the training pipeline execution
    model = joblib.load('./models/model.pkl')
    
    # Load the label encoders used during training
    # These are needed to transform categorical variables in the same way as during training
    # The encoders are stored as JSON to preserve the class mappings
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

# Initialize FastAPI application with metadata
# This creates the web application with title, description, and version information
app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices based on various features",
    version="1.0.0"
)

# Pydantic model for input validation
# This ensures that the API receives properly formatted data with correct data types
# The Field validators enforce constraints like non-negative numbers
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

# Pydantic model for API response
# This defines the structure of the prediction response
# The confidence field is constrained between 0 and 1
class PredictionResponse(BaseModel):
    predicted_price: float
    confidence: float = Field(..., ge=0, le=1)
    status: str = "success"

@app.get('/')
def main():
    """
    Health check endpoint - returns basic API information
    This endpoint provides information about the API status and version
    """
    return {
        "message": "House Price Prediction System API",
        "status": "operational",
        "version": "1.0.0"
    }

@app.post('/predict', response_model=PredictionResponse)
async def predict(data: InputFeatures):
    """
    Main prediction endpoint
    Takes house features as input and returns predicted price
    
    This endpoint performs the following steps:
    1. Validates input data using Pydantic models
    2. Converts input to DataFrame format
    3. Applies label encoding to categorical variables
    4. Makes prediction using the trained model
    5. Returns prediction with confidence score
    """
    try:
        # Convert Pydantic model to dictionary for processing
        input_dict = data.dict()
        
        # Use the same data ingestion function as the training pipeline
        # This ensures consistency between training and inference
        df = ingestdata(input_dict)

        # Apply the same label encoding as used during training
        # This is crucial for maintaining consistency between training and inference
        for col, encoder in label_encoders.items():
            if col in df.columns:
                # Handle unseen categories by using the first known category
                # This prevents errors when the API receives new categories not seen during training
                if df[col].iloc[0] not in encoder.classes_:
                    df[col] = encoder.transform([encoder.classes_[0]])
                else:
                    df[col] = encoder.transform(df[col])

        # Ensure the dataframe has the same columns as the training data
        # This prevents errors if the model expects specific features
        expected_columns = model.feature_names_in_
        df = df[expected_columns]

        # Make prediction using the loaded model
        prediction = model.predict(df)
        
        # For now, using a fixed confidence score
        # In a production system, this could be calculated based on model uncertainty
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
    """
    Health check endpoint for monitoring
    Returns the status of the model and API
    This endpoint is useful for monitoring systems to check if the service is healthy
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }