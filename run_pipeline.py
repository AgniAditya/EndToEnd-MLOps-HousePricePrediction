# Main pipeline runner script
# This file orchestrates the entire ML training process from data ingestion to model evaluation
from pipeline.training_pipeline import trianingpipeline

if __name__ == "__main__":
    """
    Entry point for running the complete ML pipeline
    This script executes the entire workflow:
    1. Data ingestion from CSV file
    2. Data cleaning and preprocessing
    3. Model training
    4. Model evaluation and metrics logging
    """
    # Run the complete training pipeline
    # The pipeline is defined in pipeline/training_pipeline.py and uses ZenML for orchestration
    trianingpipeline(data="G:/My Stuff/2_ML/Projects/House Pricing/dataset/house_prices.csv")