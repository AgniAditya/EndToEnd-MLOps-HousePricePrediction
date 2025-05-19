from abc import abstractmethod
import logging
import pandas as pd
class CleanData:
    """
    Clean the Data
    """
    def __init__(self,data : str):
        self.data = data
    def cleandata(self) -> pd.DataFrame:
        logging.info(f'Cleaning the data {self.data}')
        return pd.read_csv(self.data)