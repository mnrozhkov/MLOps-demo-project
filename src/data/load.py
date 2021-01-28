import pandas as pd
from typing import Text


def load_target(path: Text) -> pd.DataFrame:
    """ Loads binary labels from feather to df
    """
    return pd.read_feather(path)


def load_features(path: Text) -> pd.DataFrame:
    """ Loads features from feather to df
    """
    return pd.read_feather(path)


def load_data(path: Text) -> pd.DataFrame:
    """ Loads data from feather to df
    """
    return pd.read_feather(path)
