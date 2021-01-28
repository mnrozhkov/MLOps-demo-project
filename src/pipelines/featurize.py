"""This script extracts features from raw data
   Can be run from both cmd line (argparse to read config.yaml) or imported as a module
   Pipelines has artifacts on the output and describe logic"""

import argparse
from typing import Text
import yaml
import pandas as pd

from src.data.load import load_target, load_data
from src.data.process import process_features

from src.utils.logging import get_logger


def featurize(config_path: Text) -> None:
    """Extract features and save it

    Args:
        config_path {Text}: path to config file

    """
    # import configs:
    config = yaml.safe_load(open(config_path))

    log_level = config['base']['log_level']
    target_processed_path = config['data_load']['target_processed']
    dataset_processed_path = config['data_load']['dataset_processed']
    features_path = config['featurize']['features_path']

    logger = get_logger("FEATURIZE", log_level)

    # 1. load labels and features df-s:
    logger.info('Load dataset')
    target_df = load_target(target_processed_path)
    features_df = load_data(dataset_processed_path)

    # 2. process features further:
    logger.info('Process dataset')
    features_df = process_features(features_df, target_df)

    # 2.1 merge processed features and labels:
    logger.info('Add target column')
    features = features_df.copy()
    features = pd.merge(
        left=features,
        right=target_df,
        how='left',
        on=['user_id', 'month']
    )

    # 2.2 process nulls in merged df:
    logger.info('Process nulls')
    features.dropna(inplace=True)

    logger.info('Save features')
    features.to_feather(features_path)
    logger.debug(f'Features path: {features_path}')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    featurize(config_path=args.config)
