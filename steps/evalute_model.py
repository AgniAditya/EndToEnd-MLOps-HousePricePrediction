from abc import abstractmethod
import logging
from sklearn.metrics import r2_score,root_mean_squared_error
from sklearn.linear_model import LinearRegression
from zenml import step

@step

def evalutemodel(model,x_test,y_test):
    """
    Evaluete the model accuracy by calculating the Matrixs
    """
    logging.info(f'Evaluating the model accuracy')