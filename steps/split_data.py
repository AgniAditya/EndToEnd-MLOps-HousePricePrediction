from abc import abstractmethod
from sklearn.model_selection import train_test_split
import pandas as pd
import logging
import numpy as np
from typing_extensions import Annotated
from typing import Tuple
from zenml import step

@step
def splitdata(data : pd.DataFrame) -> Tuple[
Annotated[pd.DataFrame,"X_train"],
Annotated[pd.DataFrame,"X_test"],
Annotated[pd.Series,"y_train"],
Annotated[pd.Series,"y_test"],]:
    """
    Split the Clean Data into training and testing part
    """
    logging.info(f'Spiliting the data into training and testing part')
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    return X_train, y_train, X_test, y_test