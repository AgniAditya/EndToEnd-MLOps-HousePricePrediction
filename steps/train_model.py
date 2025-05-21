from abc import abstractmethod
import logging
from zenml import step
from Data_analysis.model_dev import RandomForestModel
import pandas as pd

@step
def trainmodel(X_train : pd.DataFrame,Y_train : pd.DataFrame):
    """
    Train the Model on the training data
    """
    try:
        logging.info(f'Training has started')
        model = RandomForestModel()
        model.train(X_train,Y_train)
        logging.info(f'Trainig is completed')
        return model
    except Exception as e:
        logging.error(f'Error in the training process')
        raise e