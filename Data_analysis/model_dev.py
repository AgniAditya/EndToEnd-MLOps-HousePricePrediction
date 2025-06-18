# Model Development Module
# This module contains the model training implementation using Random Forest Regressor
# It follows the Strategy pattern with an abstract Model class and concrete RandomForestModel implementation
import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

class Model:
    """
    Abstract base class defining the interface for model implementations
    
    This class follows the Strategy pattern, allowing different model types
    to be easily swapped in and out of the pipeline. All concrete model
    implementations must inherit from this class and implement the train method.
    """
    def train(self):
        """
        Abstract method that must be implemented by concrete model classes
        
        This method should handle the training of the specific model type
        and return the trained model object.
        """
        pass


class RandomForestModel(Model):
    """
    Concrete implementation of the Model interface using Random Forest Regressor
    
    Random Forest is an ensemble learning method that constructs multiple
    decision trees during training and outputs the mean prediction of the
    individual trees. This approach helps reduce overfitting and provides
    good performance for regression tasks like house price prediction.
    """
    def train(self,X_train : pd.DataFrame,y_train : pd.Series) -> RandomForestRegressor:
        """
        Train a Random Forest Regressor on the provided training data
        
        This method initializes a Random Forest model with default hyperparameters
        and fits it to the training data. The random_state is set to 42 for
        reproducibility across different runs.
        
        Args:
            X_train (pd.DataFrame): Training features (scaled and encoded)
            y_train (pd.Series): Training target variable (house prices)
        
        Returns:
            RandomForestRegressor: Trained Random Forest model ready for predictions
        
        Raises:
            Exception: If there's an error during model training
        """
        try:
            # Initialize Random Forest Regressor with fixed random state for reproducibility
            # Default hyperparameters are used, but these could be optimized through
            # hyperparameter tuning in a production environment
            rf_model = RandomForestRegressor(random_state=42)
            
            # Fit the model to the training data
            # This trains multiple decision trees and combines their predictions
            rf_model.fit(X_train,y_train)
            
            return rf_model
        except Exception as e:
            logging.error(f'Error while training the Random Forest model: {e}')
            raise e