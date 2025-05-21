import logging
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.base import RegressorMixin
import numpy as np
from zenml import step
import pandas as pd
from typing import Tuple
from typing_extensions import Annotated

@step
def evalutemodel(model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series) -> Tuple[
    Annotated[float,"r2_socre"],
    Annotated[float,"rmse"],
    Annotated[float,"mae"],
    Annotated[float,"mape"]
]:
    """
    Evaluate the model accuracy by calculating the Matrix
    """
    try:
        logging.info(f'Evaluating the model accuracy')
        rf_model_preditctions = model.predict(X_test)
        r2_score_rf = r2_score(y_test, rf_model_preditctions)
        mae = mean_absolute_error(y_test, rf_model_preditctions)
        rmse = np.sqrt(((y_test - rf_model_preditctions) ** 2).mean())
        mape = mean_absolute_percentage_error(y_test, rf_model_preditctions)
        return r2_score_rf,rmse,mae,mape
    except Exception as e:
        logging.error("Error in evaluation")
        raise e