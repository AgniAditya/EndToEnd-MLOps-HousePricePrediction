from steps.clean_data import cleandata
from steps.ingest_data import IngestData
from steps.evalute_model import evalutemodel
from steps.train_model import trainmodel
from steps.split_data import splitdata
from zenml import pipeline

@pipeline(enable_cache=True)
def trianingpipeline(data : str):
    """
    Step by Step runing pipeline
    """
    ingested_data = IngestData(data)
    clean_data = cleandata(ingested_data)
    X_train,Y_train,x_test,y_test = splitdata(clean_data)
    trained_model = trainmodel(X_train,Y_train)
    evalutemodel(trained_model,x_test,y_test)