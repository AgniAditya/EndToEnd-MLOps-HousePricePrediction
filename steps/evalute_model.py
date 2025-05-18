from abc import abstractmethod

class EvaluteModel:
    """
    Evaluete the model accuracy by calculating the Matrixs
    """
    @abstractmethod
    def evalutemodel(self,X_train,Y_train,x_test,y_test):
        pass