from steps.clean_data import clean_df
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
    X_train,X_test,y_train,y_test = clean_df(dataframe)
    trained_model = trainmodel(X_train,y_train)
    evalutemodel(trained_model,X_test,y_test)