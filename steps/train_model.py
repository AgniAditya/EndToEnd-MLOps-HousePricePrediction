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
    Train the Model on the training data and save it
    """
    try:
        logging.info(f'Training has started')
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Train the model
        object = RandomForestModel()
        model = object.train(X_train, Y_train)
        
        # Save the model locally first
        model_path = os.path.join("models", "model.pkl")
        joblib.dump(model, model_path)
        logging.info(f'Model saved to {model_path}')
        
        # Save the scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        scaler_path = os.path.join("models", "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        logging.info(f'Scaler saved to {scaler_path}')
        
        # Verify files exist
        if not os.path.exists(model_path):
            raise Exception(f"Model file was not created at {model_path}")
        if not os.path.exists(scaler_path):
            raise Exception(f"Scaler file was not created at {scaler_path}")
            
        # Log to MLflow
        try:
            with mlflow.start_run():
                mlflow.log_params({
                    "model_type": "RandomForestRegressor",
                    "random_state": 42
                })
                
                # Log the model files
                mlflow.log_artifact(model_path)
                mlflow.log_artifact(scaler_path)
                
                # Log the model
                mlflow.sklearn.log_model(
                    model,
                    "house_price_model",
                    registered_model_name="house_price_model"
                )
        except Exception as mlflow_error:
            logging.warning(f"MLflow logging failed: {str(mlflow_error)}")
            logging.info("Continuing with local model only")
        
        logging.info(f'Training is completed and model is saved locally')
        return model
    except Exception as e:
        logging.error(f'Error in the training process: {str(e)}')
        raise e