"""This script loads and pre-process raw data
   Can be run from both cmd line (argparse to read config.yaml) or imported as a module
   Pipelines has artifacts on the output and describe logic"""

import argparse
from typing import Text
import yaml

from src.data.load import load_target, load_data
from src.data.process import process_target, process_data
from src.utils.logging import get_logger


def data_load(config_path: Text) -> None:
    """Load and process data

    Args:
        config_path {Text}: path to yaml config file

    """
    # import configs:
    config = yaml.safe_load(open(config_path))

    log_level = config['base']['log_level']
    target_raw_path = config['data_load']['target']
    dataset_raw_path = config['data_load']['dataset']
    target_processed_path = config['data_load']['target_processed']
    dataset_processed_path = config['data_load']['dataset_processed']

    logger = get_logger("DATA_LOAD", log_level)

    # 1. load labels and features df-s:
    logger.info('Load dataset')
    target_df = load_target(target_raw_path)
    data_df = load_data(dataset_raw_path)

    # 2. process labels:
    logger.info('Process target')
    target_df = process_target(target_df)

    # 3. process features:
    logger.info('Process dataset')
    data_df = process_data(data_df)
    data_df.dropna(inplace=True)

    # 4. save labels and features:
    logger.info('Save processed data and target')
    target_df.to_feather(target_processed_path)
    data_df.to_feather(dataset_processed_path)
    logger.debug(f'Processed data path: {dataset_processed_path}')
    logger.debug(f'Processed data path: {target_processed_path}')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_load(config_path=args.config)
