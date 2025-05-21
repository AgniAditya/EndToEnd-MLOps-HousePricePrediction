from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
import logging

def setup_mlflow():
    """Registers and activates MLflow experiment tracker with ZenML."""
    client = Client()
    try:
        client.get_stack_component("mlflow_tracker", "experiment_tracker")
        logging.info("MLflow tracker already registered.")
    except KeyError:
        tracker = MLFlowExperimentTracker(
            name="mlflow_tracker",
            tracking_uri="http://localhost:5000"
        )
        client.register_stack_component(tracker)
        logging.info("✅ MLflow tracker registered.")

    # Set it as the active experiment tracker
    client.active_stack.experiment_tracker = "mlflow_tracker"
    logging.info("✅ MLflow tracker set as active in the current stack.")