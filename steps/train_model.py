from abc import abstractmethod
import logging
from sklearn.linear_model import LinearRegression

class TrainModel:
    """
    Train the Model on the training data
    """
    @abstractmethod
    def trainmodel(X_train,Y_train):
        logging.info(f'Training the model on data')
        model = LinearRegression()
        model.fit(X_train,Y_train)
        return model