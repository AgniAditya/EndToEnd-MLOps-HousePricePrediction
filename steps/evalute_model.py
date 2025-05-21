import logging
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import mlflow
from zenml import step

@step
def evalutemodel(model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series) -> None:
    """
    Evaluate the model accuracy by calculating the Matrix
    """
    try:
        logging.info('Evaluating model performance...')

        rf_model_preditctions = model.predict(X_test)

        r2_score_rf = r2_score(y_test, rf_model_preditctions)
        mae = mean_absolute_error(y_test, rf_model_preditctions)
        rmse = np.sqrt(((y_test - rf_model_preditctions) ** 2).mean())
        mape = mean_absolute_percentage_error(y_test, rf_model_preditctions)

        with mlflow.start_run(nested=True):
            mlflow.log_metric("r2_score", r2_score_rf)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mape", mape)

        logging.info("Metrics logged to MLflow.")
    except Exception as e:
        logging.error("Error in evaluation", exc_info=True)
        raise e