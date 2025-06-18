# ZenML Pipeline for House Price Prediction
# This file defines the complete ML workflow using ZenML's pipeline decorator
# The pipeline ensures reproducibility and tracks all steps automatically
from steps.clean_data import clean_df
from steps.ingest_data import ingestdata_step
from steps.evalute_model import evalutemodel
from steps.train_model import trainmodel
from zenml import pipeline

@pipeline(enable_cache=True)
def trianingpipeline(data : str):
    """
    Complete ML training pipeline orchestrated by ZenML
    
    This pipeline executes the following steps in sequence:
    1. Data Ingestion: Load data from CSV file
    2. Data Cleaning: Preprocess and prepare data for training
    3. Model Training: Train the Random Forest model
    4. Model Evaluation: Calculate and log performance metrics
    
    Args:
        data (str): Path to the CSV file containing house price data
    
    Returns:
        None: The pipeline saves the trained model and logs metrics to MLflow
    """
    # Step 1: Load and ingest the raw data from CSV file
    dataframe = ingestdata_step(data)
    
    # Step 2: Clean, preprocess data and split into train/test sets
    # This step handles missing values, encoding categorical variables, and data scaling
    X_train,X_test,y_train,y_test = clean_df(dataframe)
    
    # Step 3: Train the Random Forest model on the prepared training data
    trained_model = trainmodel(X_train,y_train)
    
    # Step 4: Evaluate the model performance on test data
    # This logs metrics like R², MAE, RMSE to MLflow for tracking
    evalutemodel(trained_model,X_test,y_test)