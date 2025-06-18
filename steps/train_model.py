# Model Training Step
# This module handles the training of the Random Forest model and saves it for later use
# It also integrates with MLflow for experiment tracking and model versioning
from abc import abstractmethod
import logging
from zenml import step
from Data_analysis.model_dev import RandomForestModel
import pandas as pd
import mlflow
import joblib
import os
from sklearn.preprocessing import StandardScaler

@step
def trainmodel(X_train: pd.DataFrame, Y_train: pd.DataFrame):
    """
    ZenML step for training the Random Forest model
    
    This step performs the following operations:
    1. Trains a Random Forest Regressor on the prepared training data
    2. Saves the trained model locally for API inference
    3. Saves a StandardScaler for feature scaling during inference
    4. Logs the model and artifacts to MLflow for experiment tracking
    
    The step is decorated with @step to enable caching and automatic tracking
    of training parameters and model artifacts in ZenML.
    
    Args:
        X_train (pd.DataFrame): Training features (scaled and encoded)
        Y_train (pd.DataFrame): Training target variable (house prices)
    
    Returns:
        RandomForestRegressor: Trained model object
    
    Raises:
        Exception: If there's an error during model training or saving
    """
    try:
        logging.info('Starting model training process')
        
        # Create models directory if it doesn't exist
        # This ensures the directory structure is ready for saving model files
        os.makedirs("models", exist_ok=True)
        
        # Step 1: Train the Random Forest model
        # The RandomForestModel class handles the actual training process
        object = RandomForestModel()
        model = object.train(X_train, Y_train)
        
        # Step 2: Save the trained model locally for API inference
        # This file will be loaded by the FastAPI application
        model_path = os.path.join("models", "model.pkl")
        joblib.dump(model, model_path)
        logging.info(f'Model saved successfully to {model_path}')
        
        # Step 3: Save the StandardScaler for consistent feature scaling
        # This ensures the same scaling is applied during inference as during training
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        scaler_path = os.path.join("models", "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        logging.info(f'StandardScaler saved successfully to {scaler_path}')
        
        # Step 4: Verify that model files were created successfully
        # This prevents issues where the API tries to load non-existent files
        if not os.path.exists(model_path):
            raise Exception(f"Model file was not created at {model_path}")
        if not os.path.exists(scaler_path):
            raise Exception(f"Scaler file was not created at {scaler_path}")
            
        # Step 5: Log model and artifacts to MLflow for experiment tracking
        # This enables model versioning and performance comparison across runs
        try:
            with mlflow.start_run():
                # Log training parameters for reproducibility
                mlflow.log_params({
                    "model_type": "RandomForestRegressor",
                    "random_state": 42
                })
                
                # Log model files as artifacts
                mlflow.log_artifact(model_path)
                mlflow.log_artifact(scaler_path)
                
                # Register the model in MLflow model registry
                mlflow.sklearn.log_model(
                    model,
                    "house_price_model",
                    registered_model_name="house_price_model"
                )
        except Exception as mlflow_error:
            # If MLflow logging fails, continue with local model only
            # This ensures the pipeline doesn't fail if MLflow is not available
            logging.warning(f"MLflow logging failed: {str(mlflow_error)}")
            logging.info("Continuing with local model only")
        
        logging.info('Model training completed successfully')
        return model
    except Exception as e:
        logging.error(f'Error in the training process: {str(e)}')
        raise e