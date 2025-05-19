from abc import abstractmethod
import pandas as pd
import logging
class IngestData:
    """
    Ingesting the data
    """
    def __init__(self,data : str):
        self.data = data
    
    def getData(self):
        logging.info(f'Ingesting data from {self.data_path}')
        return pd.read_csv(self.data)