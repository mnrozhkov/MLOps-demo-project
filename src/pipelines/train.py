"""This script trains the model
   Can be run from both cmd line (argparse to read config.yaml) or imported as a module
   Pipelines has artifacts on the output and describe logic"""

import argparse
import os

import catboost as ctb
import joblib
import json
import matplotlib.pyplot as plt
import pandas as pd
from typing import Text
import yaml
from envyaml import EnvYAML

from src.train.train import custom_ts_split, get_split_data
from src.utils.logging import get_logger
from src.evaluate.metrics import precision_at_k_score, recall_at_k_score, lift_score


def train(config_path: Text) -> None:
    """Train model

    Params:
        config_path {Text}: path to config file

    """
    # Import configs:
    # -------------------------------------------
    # config = yaml.safe_load(open(config_path))
    config = EnvYAML(config_path)
    print(config.get('base'))

    # base params:
    random_state = config['base']['random_state']
    log_level = config['base']['log_level']

    # features params:
    features_path = config['featurize']['features_path']
    categories = config['featurize']['categories']

    # train params:
    estimator_params = config['train']['catboost_params']
    # estimator = config['train']['estimator']
    top_K_coef = config['train']['top_K_coef']
    model_path = config['train']['model_path']
    raw_metrics_path = config['train']['raw_metrics_path']
    train_metrics_path = config['train']['train_metrics_path']
    train_plots_path = config['train']['train_plots_path']
    train_metrics_png = config['train']['train_metrics_png']
    # -------------------------------------------

    logger = get_logger("TRAIN", log_level)

    # 1. load and process joint data:
    logger.info('Load data')
    features = pd.read_feather(features_path)
    features['month'] = pd.to_datetime(features['month'])

    # 2. instantiate a model:
    # logger.info(f'Estimator = {estimator}')
    logger.info('Instantiate model')
    clf = ctb.CatBoostClassifier(
        **estimator_params,
        cat_features=categories,
        random_state=random_state
    )

    # 3. count top k instances for subsequent train and evaluations
    metrics_df = pd.DataFrame(columns=['test_period', 'lift', 'precision_at_k', 'recall_at_k'])
    top_K = int(features.shape[0] * top_K_coef)
    months = features.month.sort_values().unique()
    logger.info(f'Top_K {top_K_coef*100}% of the dataset size: {top_K}')

    # 4. train model
    k = 1
    for start_train, end_train, test_period in custom_ts_split(months, train_period=1):

        logger.info(f'Fold {k}:')
        logger.info(f'Train: {start_train} - {end_train}')
        logger.info(f'Test: {test_period} \n')

        # Get train / test data for the split
        X_train, X_test, y_train, y_test = get_split_data(features, start_train, end_train, test_period)
        logger.info(f'Train shapes: X - {X_train.shape}, y - {y_train.shape}')
        logger.info(f'Test shapes: X - {X_test.shape}, y - {y_test.shape}')

        # Fit estimator
        clf.fit(X_train, y_train)

        # Predict on test
        y_pred = clf.predict(X_test)
        probas = clf.predict_proba(X_test)
        logger.info(f'Max probas: {probas[:, 1].max()}')

        # Calculate raw metrics on test per each fold:
        # -------------------------------------------
        lift = lift_score(y_test, y_pred, probas[:, 1], top_K)
        precision_at_k = precision_at_k_score(y_test, y_pred, probas[:, 1], top_K)
        recall_at_k = recall_at_k_score(y_test, y_pred, probas[:, 1], top_K)
        metrics_df = metrics_df.append(
            dict(zip(metrics_df.columns, [test_period, lift, precision_at_k, recall_at_k])),
            ignore_index=True
            )

        k += 1
        logger.info(f'Precision at {top_K}: {precision_at_k}')
        logger.info(f'Recall at {top_K}: {recall_at_k}\n')

    logger.info('Save "raw" metrics for plotting')
    metrics_df.to_csv(raw_metrics_path, index=False)

    # Create and safe aggregated (min, max, std, mean) metrics:
    # -------------------------------------------
    logger.info('Save aggregated metrics')
    metrics_aggs = metrics_df[['lift', 'precision_at_k', 'recall_at_k']].agg(['max', 'min', 'std', 'mean'])
    metrics = {
        f'{metric}_{agg}': metrics_aggs.loc[agg, metric]
        for metric in metrics_aggs.columns
        for agg in metrics_aggs.index
    }
    with open(os.path.join(config['base']['project_dir'], train_metrics_path), 'w') as metrics_f:
        json.dump(obj=metrics, fp=metrics_f, indent=4)

    # Generate and save data for plots:
    # -------------------------------------------
    logger.info('Generate & save plots')
    plots_df = pd.DataFrame({
        'metric': list(metrics.keys()),
        'value': list(metrics.values())
    })
    plots_df.to_csv(train_plots_path, index=False)

    # Make a plot using a csv above:
    x_labels = list(range(0, len(metrics)))
    plt.figure(figsize=(10, 10))
    fig = plt.bar(x_labels, list(metrics.values()))
    plt.title('Train metrics', fontsize=14)
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.xticks(x_labels, metrics.keys(), size='small', rotation='45')
    plt.grid(color='k', linestyle='-', linewidth=0.5)
    # plt.show()
    plt.savefig(train_metrics_png)

    # Save the trained model:
    # -------------------------------------------
    logger.info('Save model')
    path = os.path.join(config['base']['project_dir'], model_path)
    joblib.dump(clf, path)
    logger.info(f'Model saved to: {path}')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train(config_path=args.config)
