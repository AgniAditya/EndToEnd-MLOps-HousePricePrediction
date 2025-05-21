from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
import logging
from zenml.enums import StackComponentType

def setup_mlflow():
    """Registers and activates MLflow experiment tracker with ZenML."""
    client = Client()

    active_stack = client.active_stack
    try:
        client.get_stack_component("mlflow_tracker", StackComponentType.EXPERIMENT_TRACKER)
        logging.info("MLflow tracker already registered.")
    except KeyError:
        tracker = MLFlowExperimentTracker(
            name="mlflow_tracker",
            tracking_uri="http://localhost:5000"
        )
        active_stack.experiment_tracker(tracker)
        logging.info("✅ MLflow tracker registered.")

    # Set it as the active experiment tracker
    active_stack.experiment_tracker = "mlflow_tracker"
    logging.info("✅ MLflow tracker set as active in the current stack.")