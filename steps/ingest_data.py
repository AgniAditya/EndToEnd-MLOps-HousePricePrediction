import pandas as pd
import logging
from zenml import step


class IngestData:
    """
    Ingesting the data
    """
    def __init__(self,data : str):
        self.data = data
     
    def __init__(self,data : dict):
        self.data = data
    
    def getData(self):
        logging.info(f'Ingesting data from {self.data}')
        return pd.DataFrame([self.data])

@step
def ingestdata(data : str) -> pd.DataFrame:
    """
    Ingesting the data

    Args:
        data is the string of the data path
    Returns:
        It retruns pandas DataFrame
    """

    try:
        return IngestData(data).getData()
    except Exception as e:
        logging.error(f'Error while ingesting the data: {e}')
        raise e

def ingestdata(data : dict) -> pd.DataFrame:
    """
    Ingest the data

    Args:
        data : Dictonary format
    Returns:
        Pandas Dataframe
    """

    try :
        return IngestData(data).getData()
    except Exception as e:
        logging.error(f'Error while ingesting the data: {e}')
        raise e