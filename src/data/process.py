import pandas as pd


def process_features(features_df, target_df):
    """Join features with labels

    """
    features_df = features_df.loc[features_df.user_id.isin(target_df.user_id)]
    features_df['month'] = pd.to_datetime(features_df['month'])

    return features_df


def process_data(df):
    """Convert 'month' feature to datetime

    """
    df['month'] = pd.to_datetime(df['month'])

    return df


def process_target(df):
    """Convert 'month' feature to datetime

    """
    df['month'] = pd.to_datetime(df['month'])

    return df


def join_target_and_features(target_df, user_features_df):
    """Join processed features with labels in one common df

    """
    features = user_features_df.copy()
    features = pd.merge(
        left=features,
        right=target_df,
        how='left',
        on=['user_id', 'month']
    )
    features.dropna(inplace=True)

    return features
