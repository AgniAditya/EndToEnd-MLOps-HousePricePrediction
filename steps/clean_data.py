from abc import abstractmethod
import logging
import pandas as pd
from zenml import step

@step
def cleandata(data) -> pd.DataFrame:
    """
    Clean the Data
    """
    logging.info(f'Cleaning the data {data}')
    return pd.read_csv(data)