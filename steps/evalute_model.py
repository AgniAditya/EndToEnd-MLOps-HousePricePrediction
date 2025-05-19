from abc import abstractmethod
import logging
from sklearn.metrics import r2_score,root_mean_squared_error
from sklearn.linear_model import LinearRegression

class EvaluteModel:
    """
    Evaluete the model accuracy by calculating the Matrixs
    """
    @abstractmethod
    def evalutemodel(model,X_train,Y_train,x_test,y_test):
        logging.info(f'Evaluating the model accuracy')