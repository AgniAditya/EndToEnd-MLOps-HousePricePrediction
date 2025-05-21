import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
class Model:
    """
    Interface of the Model implementation
    """
    def train(self):
        pass


class RandomForestModel(Model):
    """
    Implementation of the RandomForestModel
    """
    def train(self,X_train : pd.DataFrame,y_train : pd.Series) -> RandomForestRegressor:
        try:
            rf_model = RandomForestRegressor(random_state=42)
            rf_model.fit(X_train,y_train)
            return rf_model
        except Exception as e:
            logging.error(f'Error while training the model')
            raise e