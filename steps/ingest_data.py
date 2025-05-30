import pandas as pd
import logging
from zenml import step


class IngestData:
    """
    Ingesting the data
    """
    def __init__(self, data: dict):
        self.data = data
    
    def getData(self):
        logging.info('Ingesting data from dictionary')
        # Convert dictionary to DataFrame
        df = pd.DataFrame([self.data])
        # Ensure all columns are present
        required_columns = ['Title', 'Bathroom', 'Carpet Area', 'location', 'Transaction', 
                          'Furnishing', 'Balcony', 'facing', 'Price (in rupees)',
                          'Status', 'Society', 'Floor']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        return df

# This version is for ZenML pipeline
@step
def ingestdata_step(data: str) -> pd.DataFrame:
    """
    Ingesting the data for ZenML pipeline

    Args:
        data: string path to the data file
    Returns:
        pandas DataFrame
    """
    try:
        return pd.read_csv(data)
    except Exception as e:
        logging.error(f'Error while ingesting the data: {e}')
        raise e

# This version is for API
def ingestdata(data: dict) -> pd.DataFrame:
    """
    Ingest the data for API predictions

    Args:
        data: Dictionary containing the input features
    Returns:
        pandas DataFrame
    """
    try:
        return IngestData(data).getData()
    except Exception as e:
        logging.error(f'Error while ingesting the data: {e}')
        raise e