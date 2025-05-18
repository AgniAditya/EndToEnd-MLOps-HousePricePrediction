from abc import abstractmethod

class SplitData:
    """
    Split the Clean Data into training and testing part
    """
    @abstractmethod
    def splitdata(self,data):
        pass