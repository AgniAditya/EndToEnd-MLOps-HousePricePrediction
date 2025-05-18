from abc import abstractmethod

class TrainModel:
    """
    Train the Model on the training data
    """
    @abstractmethod
    def trainmodel(self,X_train,Y_train):
        pass