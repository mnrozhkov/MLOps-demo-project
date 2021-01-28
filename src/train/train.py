"""Functions for training"""
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd

from src.utils.logging import get_logger
import logging

from typing import Iterator, Tuple


def custom_ts_split(months,
                    train_period: int = 0) -> Iterator:
    """ Custom data split for time-based learning

    Args:
        months: {np.array}array of months
        train_period: {int} train period duration, in months

    Yields:
       {Iterator: Tuple[pd.DataFrame]}: start_train, end_train, test_period

    """
    logger = get_logger(__name__, logging.INFO)

    for k, month in enumerate(months):
        start_train = pd.to_datetime(months.min())
        end_train = start_train + MonthEnd(train_period + k-1)
        test_period = pd.to_datetime(end_train + MonthEnd(1))

        if test_period <= pd.to_datetime(months.max()):
            yield start_train, end_train, test_period
        else:
            # print(test_period)
            # print(months.max())
            logger.info(test_period)
            logger.info(months.max())


def get_split_data(features: pd.DataFrame,
                   start_train: pd.Timestamp,
                   end_train: pd.Timestamp,
                   test_period: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Having a tuple from custom_ts_split make a split of features accordingly

    Args:
        features: {pd.DataFrame} original dataframe of features
        start_train: {pd.Timestamp} starting month in a train sample
        end_train: {pd.Timestamp} ending month in a train sample
        test_period: {pd.Timestamp} test month

    Returns: {Tuple[pd.DataFrame]} x_train, x_test, y_train, y_test

    """

    # Get train and test data for the split
    x_train = (features[(features.month >= start_train) & (features.month <= end_train)]
               .drop(columns=['user_id', 'month', 'target'], axis=1))
    x_test = (features[(features.month == test_period)]
              .drop(columns=['user_id', 'month', 'target'], axis=1))

    y_train = features.loc[(features.month >= start_train) & (features.month <= end_train), 'target']
    y_test = features.loc[(features.month == test_period), 'target']

    return x_train, x_test, y_train, y_test
