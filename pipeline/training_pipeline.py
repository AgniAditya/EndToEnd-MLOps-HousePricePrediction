from steps.clean_data import cleandata
from steps.ingest_data import ingestdata
from steps.evalute_model import evalutemodel
from steps.train_model import trainmodel
from zenml import pipeline

@pipeline(enable_cache=True)
def trianingpipeline(data : str):
    """
    Step by Step runing pipeline
    """
    dataframe = ingestdata(data)
    X_train,Y_train,x_test,y_test = cleandata(dataframe)
    trained_model = trainmodel(X_train,Y_train)
    evalutemodel(trained_model,x_test,y_test)