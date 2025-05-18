from steps.clean_data import CleanData
from steps.ingest_data import IngestData
from steps.evalute_model import EvaluteModel
from steps.train_model import TrainModel
from steps.split_data import SplitData


def trianingpipeline(data : str):
    """
    Step by Step runing pipeline
    """
    IngestData.ingestdata(data)
    clean_data = CleanData.cleandata(data)
    X_train,Y_train,x_test,y_test = SplitData.splitdata(clean_data)
    trained_model = TrainModel.trainmodel(X_train,Y_train)
    EvaluteModel.evalutemodel(trained_model,X_train,Y_train,x_test,y_test)