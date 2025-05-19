from abc import abstractmethod
from sklearn.model_selection import train_test_split
import pandas as pd
import logging
import numpy as np
from typing_extensions import Annotated
from typing import Tuple

class SplitData:
    """
    Split the Clean Data into training and testing part
    """
    def splitdata(data : pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"],]:
        logging.info(f'Spiliting the data into training and testing part')
        X,y = X, y = np.arange(10).reshape((5, 2)), range(5)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test