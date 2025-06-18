# Model Evaluation Step
# This module evaluates the trained model's performance on test data
# It calculates multiple regression metrics and logs them to MLflow for tracking
import logging
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import mlflow
from zenml import step

@step(enable_cache=False)
def evalutemodel(model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series) -> None:
    """
    ZenML step for evaluating model performance on test data
    
    This step calculates multiple regression metrics to assess model performance:
    - R² Score: Coefficient of determination (how well the model explains variance)
    - MAE: Mean Absolute Error (average absolute difference between predictions and actual values)
    - RMSE: Root Mean Square Error (penalizes larger errors more heavily)
    - MAPE: Mean Absolute Percentage Error (error as a percentage of actual values)
    
    All metrics are logged to MLflow for experiment tracking and comparison.
    The step is decorated with @step(enable_cache=False) to ensure evaluation
    always runs, even if the model hasn't changed.
    
    Args:
        model (RandomForestRegressor): Trained Random Forest model
        X_test (pd.DataFrame): Test features for evaluation
        y_test (pd.Series): Actual house prices for comparison
    
    Returns:
        None: Metrics are logged to MLflow instead of being returned
    
    Raises:
        Exception: If there's an error during model evaluation
    """
    try:
        logging.info('Starting model evaluation on test data')

        # Generate predictions on test data using the trained model
        rf_model_preditctions = model.predict(X_test)

        # Calculate various regression metrics to assess model performance
        # R² Score: Measures how well the model explains the variance in the target variable
        # Values range from 0 to 1, where 1 indicates perfect prediction
        r2_score_rf = r2_score(y_test, rf_model_preditctions)
        
        # MAE: Average absolute difference between predicted and actual values
        # This metric is in the same units as the target variable (rupees)
        mae = mean_absolute_error(y_test, rf_model_preditctions)
        
        # RMSE: Square root of the mean squared error
        # This metric penalizes larger errors more heavily than MAE
        rmse = np.sqrt(((y_test - rf_model_preditctions) ** 2).mean())
        
        # MAPE: Mean absolute percentage error
        # This metric expresses error as a percentage of actual values
        # Useful for understanding relative prediction accuracy
        mape = mean_absolute_percentage_error(y_test, rf_model_preditctions)

        # Log all metrics to MLflow for experiment tracking
        # This enables comparison of model performance across different runs
        mlflow.log_metric("r2_score", r2_score_rf)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)

        logging.info("Model evaluation completed - metrics logged to MLflow successfully")
    except Exception as e:
        logging.error("Error during model evaluation", exc_info=True)
        raise e