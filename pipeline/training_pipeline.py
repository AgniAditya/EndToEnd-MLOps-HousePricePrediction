from steps.clean_data import CleanData
from steps.ingest_data import IngestData
from steps.evalute_model import EvaluteModel
from steps.train_model import TrainModel
from steps.split_data import SplitData
from zenml import pipeline

@pipeline(enable_cache=True)
def trianingpipeline(data : str):
    """
    Step by Step runing pipeline
    """
    ingested_data = IngestData(data)
    clean_data = CleanData(ingested_data.data).cleandata()
    X_train,Y_train,x_test,y_test = SplitData.splitdata(clean_data)
    trained_model = TrainModel.trainmodel(X_train,Y_train)
    EvaluteModel.evalutemodel(trained_model,x_test,y_test)