from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline


def predict_proba(pipeline: Pipeline, data: pd.DataFrame) -> np.ndarray:
    """
    Predicting data with given model
    :param pipeline: model for prediction
    :param data: data for prediction
    :return: probability of positive class
    """
    return pipeline.predict_proba(data)[:, 1]


def compute_metrics(
    predicts: np.ndarray, target: pd.Series, threshold: float
) -> Dict[str, float]:
    """
    Compute metrics for given predictions and targets
    :param predicts: probabilities of positive class
    :param target: true labels
    :param threshold: threshold to convert probability into positive label
    :return: roc_auc, f1_score, accuracy_score stored in dict
    """
    return {
        "roc_auc": roc_auc_score(target, predicts),
        "f1_score": f1_score(target, predicts > threshold),
        "accuracy": accuracy_score(target, predicts > threshold),
    }
