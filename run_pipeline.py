from pipeline.training_pipeline import trianingpipeline
from config.mlflow_setup import setup_mlflow

if __name__ == "__main__":
    """
    Stepup the MLFlow
    """
    setup_mlflow()
    """
    Run the Pipeline
    """
    trianingpipeline(data="G:/My Stuff/2_ML/Projects/House Pricing/dataset/house_prices.csv")