from abc import ABC
from typing import Union
import pandas as pd


class DataStrategy(ABC):
    """
    Abstract class defining startegy for handling data
    """

    def handle_data(self,data: str) -> Union[pd.DataFrame,pd.Series]:
        pass

class DataPreprocessing(DataStrategy):
    """
    Pre process the Data for model training
    """

    def handle_data(self,data : pd.DataFrame) -> pd.DataFrame:
        pass



class DataDivision(DataStrategy):
    """
    Divides data into Training and Testing part
    """
    def handle_data(self, data: pd.DataFrame):
        pass

class DataCleaning:
    def __init__(self,data : pd.DataFrame,Strategy : DataStrategy):
        self.data = data
        self.Strategy = Strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_data(self.df)