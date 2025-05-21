import logging
import pandas as pd
from zenml import step
from Data_analysis.data_cleaning import (
    DataDivision,
    DataPreprocessing,
)
from typing_extensions import Annotated
from typing import Tuple
import numpy as np

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"],
]:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.

    Args:
        data: pd.DataFrame

    Returns:
        X_trian : Traning Data
        X_test : Testing Data
        y_trian : Traning label
        y_test : Testing labels
    """
    try:
        logging.info("Data cleaning Started")
        df = DataPreprocessing().handle_data(df)
        logging.info("Data cleaning comleted")

        logging.info("Data dividing has started")
        X_train, X_test, y_train, y_test = DataDivision().handle_data(df)
        logging.info("Data dividing has ended")

        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e