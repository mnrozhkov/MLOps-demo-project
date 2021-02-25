"""This script logs metadata to mlflow server."""

import argparse
import cb_flavor
import json
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import os
import pandas as pd
from typing import Text
import yaml

from src.utils.errors import UnknownEstimatorError
from src.utils.mlflow_utils import create_mlflow_experiment
from src.utils.logging import get_logger


def log_to_mlflow(config_path: Text) -> None:
    """
    Logs metadata to mlflow server.
    Args:
        config_path {Text}: path to config
    """
    # Import configs:
    # -------------------------------------------
    config = yaml.safe_load(open(config_path))

    # Base params
    random_state = config['base']['random_state']
    log_level = config['base']['log_level']
    exp_name = config['base']['exp_name']

    # Data & Features params
    target = config['data_load']['target']
    features_path = config['featurize']['features_path']
    categories = config['featurize']['categories']

    # Train params
    # available_estimators = config['train']['estimators']
    estimator = config['train']['estimator']
    estimator_params = config['train']['catboost_params']
    top_K_coef = config['train']['top_K_coef']
    model_path = config['train']['model_path']
    raw_metrics = config['train']['raw_metrics_path']
    train_metrics = config['train']['train_metrics_path']
    train_metrics_png = config['train']['train_metrics_png']

    mlflow_report = config['log_metrics']['mlflow_report_path']
    # -------------------------------------------

    logger = get_logger('LOG_METRICS', log_level)
    logger.info('Start logging')

    # client = MlflowClient(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
    #                       registry_uri=os.environ.get("MLFLOW_STORAGE"))

    client = MlflowClient()  # create MLflow client with default local uri:s

    # Create a new experiment + a new run:
    exp_id = create_mlflow_experiment(exp_name)
    run = client.create_run(experiment_id=exp_id, start_time=None, tags=None)
    client.set_experiment_tag(exp_id, "Models packages ", "CatBoost")
    logger.debug(f'Experiment ID: {exp_id}, Run ID: {run.info.run_id}')

    with mlflow.start_run(run_id=run.info.run_id, experiment_id=exp_id): # === MLflow ===
        logger.info('Log train metrics')

        with open(train_metrics) as tm_file:
            metrics = json.load(tm_file)
            mlflow.log_metrics(metrics)  # === MLflow ===
        logger.info('Log cross-validation metrics (raw metrics, by folds')

        # (timestamp - int in milliseconds)
        raw_metrics_df = pd.read_csv(raw_metrics)
        for metric, values in raw_metrics_df.set_index('test_period').to_dict().items():
            logger.debug(f'{metric}: {values}')
            for step, value in values.items():
                client.log_metric(
                    run_id=run.info.run_id,
                    key=metric,
                    value=value,
                    timestamp=int(pd.to_datetime(step).timestamp() * 1000))  # === MLflow ===

        logger.info('Log params')
        mlflow.log_params({
            'random_state': random_state,
            'categories': json.dumps(categories),
            'estimator': estimator,
            'catboost_params': json.dumps(estimator_params),
            'top_K_coef': top_K_coef
        })  # === MLflow ===

        logger.info('Log artifacts')
        mlflow.log_artifact(config_path)
        mlflow.log_artifact(target)
        mlflow.log_artifact(features_path)
        mlflow.log_artifact(train_metrics_png)

        logger.info('Log model')
        model = joblib.load(model_path)

        if estimator == 'catboost':
            cb_flavor.log_model(model, 'model')
        elif estimator == 'random_forest':
            mlflow.sklearn.log_model(model, 'model')
        else:
            raise UnknownEstimatorError(f'Unknown estimator: {estimator}')

        logger.info('Log mlflow report')

        with open(mlflow_report, 'w') as mrf:
            mrf.write(f'Experiment id: {exp_id}\n\n')
            mrf.write(f'Experiment name: {exp_name}\n\n')
            mrf.write(f'Run id: {run.info.run_id}\n\n')


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    log_to_mlflow(config_path=args.config)
