import logging
import pandas as pd
from zenml import step
from Data_analysis.data_cleaning import (
    DataDivideStrategy,
    DataPreProcessing,
    DataCleaning
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
    Data cleaning class which preprocesses the data and divides it into train and test data.

    Args:
        data: pd.DataFrame

    Returns:
        X_trian : Traning Data
        X_test : Testing Data
        y_trian : Traning label
        y_test : Testing labes
    """
    try:
        logging.info("Data cleaning Started")
        process_Startegy = DataPreProcessing()
        data_cleaning = DataCleaning(df,process_Startegy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = data_cleaning(processed_data,divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning comleted")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e