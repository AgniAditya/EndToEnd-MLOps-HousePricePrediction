from pipeline.training_pipeline import trianingpipeline
from zenml import pipeline

if __name__ == "__main__":
    """
    Run the Pipeline
    """
    trianingpipeline(data="G:/My Stuff/2_ML/Projects/House Pricing/dataset/house_prices.csv")