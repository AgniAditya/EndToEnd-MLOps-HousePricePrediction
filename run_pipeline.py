import mlflow
from pipeline.training_pipeline import trianingpipeline

if __name__ == "__main__":
    """
    Run the Pipeline
    """
    # Set MLflow tracking URI to the server
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Set the experiment name
    mlflow.set_experiment("house_price_prediction")
    
    # Run the pipeline
    trianingpipeline(data="G:/My Stuff/2_ML/Projects/House Pricing/dataset/house_prices.csv")