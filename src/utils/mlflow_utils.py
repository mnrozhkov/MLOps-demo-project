import mlflow
import os
from typing import Text


def create_mlflow_experiment(experiment_name: Text, mode: oct = 0o777) -> int:
    """
    Set mlflow experiment and permissions for folder
    Args:
        experiment_name {Text}: experiment name
        mode {oct}: experiment name mode, default = 0o777
    """

    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    artifact_location = experiment.artifact_location.replace('file://', '')

    if not os.path.exists(artifact_location):
        os.mkdir(artifact_location)
        os.chmod(artifact_location, mode)  # Change the access permissions

    return experiment.experiment_id
