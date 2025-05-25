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
        object = RandomForestModel()
        model = object.train(X_train, Y_train)
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Save the model and scaler locally
        joblib.dump(model, "models/model.pkl")
        
        # Save the scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        joblib.dump(scaler, "models/scaler.pkl")
        
        # Create input example for model signature
        input_example = X_train.iloc[:1].copy()  # Take first row as example
        
        # Log and register the model with MLflow
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                "model_type": "RandomForestRegressor",
                "random_state": 42
            })
            
            # Log the model with signature and input example
            mlflow.sklearn.log_model(
                model,
                "house_price_model",
                input_example=input_example,
                registered_model_name="house_price_model"
            )
            
            # Get the latest version
            client = mlflow.tracking.MlflowClient()
            model_versions = client.search_model_versions("name='house_price_model'")
            latest_version = max([int(mv.version) for mv in model_versions])
            
            # Transition the model to Production stage
            client.transition_model_version_stage(
                name="house_price_model",
                version=latest_version,
                stage="Production"
            )
        
        logging.info(f'Training is completed and model is saved locally')
        return model
    except Exception as e:
        logging.error(f'Error in the training process: {str(e)}')
        raise e