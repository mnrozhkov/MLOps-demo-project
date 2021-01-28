import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score
from typing import Iterable

import logging
from src.utils.logging import get_logger  # our module logging.py


def plot_lift_curve(y_true: Iterable, y_pred: Iterable, step: float = 0.01) -> None:
    """ Plot a Lift curve using the real label values of a dataset
        and the probability predictions of a Machine Learning Algorithm/model

    Params:
        y_val: true labels
        y_pred: probability predictions
        step: steps in the percentiles

    Reference:
        https://towardsdatascience.com/the-lift-curve-unveiled-998851147871

    """
    # Define an auxiliary df to plot the curve
    aux_lift = pd.DataFrame()
    aux_lift['real'] = y_true
    aux_lift['predicted'] = y_pred
    aux_lift.sort_values('predicted', ascending=False, inplace=True)

    x_val = np.arange(step, 1 + step, step)  # values on the X axis of our plot
    ratio_ones = aux_lift['real'].sum() / len(aux_lift)  # ratio of ones in our data
    y_v = []  # empty vector with the values that will go on the Y axis our our plot

    # for each x value calculate its corresponding y value
    for x in x_val:
        num_data = int(np.ceil(x * len(aux_lift)))
        data_here = aux_lift.iloc[:num_data, :]
        ratio_ones_here = data_here['real'].sum() / len(data_here)
        y_v.append(ratio_ones_here / ratio_ones)

    # plot the figure
    fig, axis = plt.subplots()
    fig.figsize = (40, 40)
    axis.plot(x_val, y_v, 'g-', linewidth=3, markersize=5)
    axis.plot(x_val, np.ones(len(x_val)), 'k-')
    axis.set_xlabel('Proportion of sample')
    axis.set_ylabel('Lift')
    plt.title('Lift Curve')
    plt.show()

    return None


def precision_at_k_score(actual: Iterable, predicted: Iterable, predicted_probas: Iterable, k: int) -> float:
    """ Precision @k metric.

    Params:
        actual {Iterable}: actual labels of the data
        predicted {Iterable}: predicted labels of the data
        predicted_probas {Iterable}: probability predictions for such data
        k {int}: top k value on which score will be calculated

    Returns:
        float: Precision@k value

    """
    df = pd.DataFrame({'actual': actual, 'predicted': predicted, 'probas': predicted_probas})
    df = df.sort_values(by=['probas'], ascending=False).reset_index(drop=True)
    df = df[:k]

    return precision_score(df['actual'], df['predicted'])


def recall_at_k_score(actual: Iterable, predicted: Iterable, predicted_probas: Iterable, k: int) -> float:
    """ Recall@k metric.

    Params:
        actual {Iterable}: actual labels of the data
        predicted {Iterable}: predicted labels of the data
        predicted_probas {Iterable}: probability predictions for such data
        k {int}: top k value on which score will be calculated

    Returns:
       float: Recall@k value

    """
    df = pd.DataFrame({'actual': actual, 'predicted': predicted, 'probas': predicted_probas})
    df = df.sort_values(by=['probas'], ascending=False).reset_index(drop=True)
    df = df[:k]

    return recall_score(df['actual'], df['predicted'])


def lift_score(actual: Iterable, predicted: Iterable, predicted_probas: Iterable, k: int) -> float:
    """ Lift@k metric.

        Params:
            actual {Iterable}: actual labels of the data
            predicted {Iterable}: predicted labels of the data
            predicted_probas {Iterable}: probability predictions for such data
            k {int}: top k value on which score will be calculated

        Returns:
            float: Lift@k value

    """

    numerator = precision_at_k_score(actual, predicted, predicted_probas, k)
    denominator = np.mean(actual)
    lift = numerator / denominator

    # print(f'Lift: {numerator} / {denominator} = {lift}')  # replaced with log:
    logger = get_logger(__name__, logging.INFO)
    logger.info(f'Lift: {numerator} / {denominator} = {lift}')

    return lift
