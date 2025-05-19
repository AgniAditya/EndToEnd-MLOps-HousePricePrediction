from abc import abstractmethod
import logging
from sklearn.linear_model import LinearRegression
from zenml import step

@step
def trainmodel(X_train,Y_train):
    """
    Train the Model on the training data
    """
    logging.info(f'Training the model on data')
    model = LinearRegression()
    model.fit(X_train,Y_train)
    return model