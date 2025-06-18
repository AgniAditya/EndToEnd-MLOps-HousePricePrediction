# Data Ingestion Module
# This module handles loading data from different sources (CSV files and API requests)
# It provides two interfaces: one for the ZenML pipeline and one for the FastAPI application
import pandas as pd
import logging
from zenml import step


class IngestData:
    """
    Data ingestion class for handling dictionary input from API requests
    
    This class converts API input data (dictionary format) into a pandas DataFrame
    that matches the structure expected by the model training pipeline.
    """
    def __init__(self, data: dict):
        """
        Initialize with input data dictionary
        
        Args:
            data (dict): Dictionary containing house features from API request
        """
        self.data = data
    
    def getData(self):
        """
        Convert dictionary input to DataFrame format
        
        This method ensures all required columns are present in the DataFrame,
        even if they're not provided in the input dictionary. This maintains
        consistency with the training data structure.
        
        Returns:
            pd.DataFrame: DataFrame with all required columns for model prediction
        """
        logging.info('Converting API input dictionary to DataFrame')
        # Convert dictionary to DataFrame with single row
        df = pd.DataFrame([self.data])
        
        # Ensure all required columns are present (same as training data)
        # If a column is missing, it will be filled with None
        required_columns = ['Title', 'Bathroom', 'Carpet Area', 'location', 'Transaction', 
                          'Furnishing', 'Balcony', 'facing', 'Price (in rupees)',
                          'Status', 'Society', 'Floor']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        return df

# ZenML Step for Pipeline Data Ingestion
@step
def ingestdata_step(data: str) -> pd.DataFrame:
    """
    ZenML step for ingesting data from CSV file during training pipeline
    
    This function is used by the ZenML pipeline to load the training dataset
    from a CSV file. It's decorated with @step to enable caching and tracking.
    
    Args:
        data (str): File path to the CSV file containing house price data
    
    Returns:
        pd.DataFrame: Raw dataset loaded from CSV file
    
    Raises:
        Exception: If there's an error reading the CSV file
    """
    try:
        logging.info(f'Loading data from CSV file: {data}')
        return pd.read_csv(data)
    except Exception as e:
        logging.error(f'Error while ingesting the data: {e}')
        raise e

# API Data Ingestion Function
def ingestdata(data: dict) -> pd.DataFrame:
    """
    Function for ingesting API request data during prediction
    
    This function is used by the FastAPI application to convert incoming
    JSON requests into the DataFrame format expected by the trained model.
    
    Args:
        data (dict): Dictionary containing house features from API request
    
    Returns:
        pd.DataFrame: DataFrame formatted for model prediction
    
    Raises:
        Exception: If there's an error processing the input data
    """
    try:
        logging.info('Processing API input data for prediction')
        return IngestData(data).getData()
    except Exception as e:
        logging.error(f'Error while ingesting the data: {e}')
        raise e