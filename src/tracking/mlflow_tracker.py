"""
MLflow wrapper for experiment tracking.
Logs params, metrics, artifacts, and models for every pipeline run.
"""
import os
from pathlib import Path

import mlflow


class MLflowTracker:
    def __init__(self, experiment_name, tracking_uri=None, artifact_location=None):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "sqlite:///mlflow.db"
        self.artifact_location = artifact_location

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name=None):
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params):
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

    def log_artifact(self, local_path):
        if Path(local_path).exists():
            mlflow.log_artifact(str(local_path))

    def log_model(self, model, model_name):
        mlflow.sklearn.log_model(
            model,
            artifact_path=model_name,
            serialization_format="cloudpickle"
        )

    def end_run(self):
        mlflow.end_run()