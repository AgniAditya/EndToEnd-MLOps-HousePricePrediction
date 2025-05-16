from steps import clean_data
from steps import ingest_data
from steps import evalute_model
from steps import train_model
from steps import split_data

def Trianingpipeline(data):
    ingest_data.IngestData(data)
    cleandata = clean_data.CleanData(data)
    X_train,Y_train,x_test,y_test = split_data.splitData(cleandata)
    trainedmodel = train_model.TrainModel(X_train,Y_train)
    evalute_model.EvaluteModel(trainedmodel,X_train,Y_train,x_test,y_test)