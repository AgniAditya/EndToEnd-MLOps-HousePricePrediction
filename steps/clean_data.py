# Data Cleaning and Preprocessing Step
# This module handles data preprocessing and train-test splitting for the ML pipeline
# It uses the Data_analysis module classes to clean data and prepare it for model training
import logging
import pandas as pd
from zenml import step
from Data_analysis.data_cleaning import (
    DataDivision,
    DataPreprocessing,
)
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"],
]:
    """
    ZenML step for cleaning and preprocessing the dataset
    
    This step performs two main operations:
    1. Data Preprocessing: Clean the raw data by handling missing values,
       removing duplicates, encoding categorical variables, and scaling features
    2. Data Division: Split the cleaned data into training and testing sets
    
    The step is decorated with @step to enable caching and automatic tracking
    of the preprocessing operations in ZenML.
    
    Args:
        df (pd.DataFrame): Raw dataset loaded from CSV file
    
    Returns:
        Tuple containing:
        - X_train (pd.DataFrame): Training features
        - X_test (pd.DataFrame): Testing features  
        - y_train (pd.Series): Training target variable (house prices)
        - y_test (pd.Series): Testing target variable (house prices)
    
    Raises:
        Exception: If there's an error during data cleaning or splitting
    """
    try:
        # Step 1: Clean and preprocess the raw data
        # This handles missing values, duplicates, categorical encoding, and feature scaling
        logging.info("Starting data cleaning and preprocessing")
        df = DataPreprocessing().handle_data(df)
        logging.info("Data cleaning and preprocessing completed successfully")

        # Step 2: Split the cleaned data into training and testing sets
        # This creates the datasets needed for model training and evaluation
        logging.info("Starting data division into train/test sets")
        X_train, X_test, y_train, y_test = DataDivision().handle_data(df)
        logging.info("Data division completed successfully")

        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e