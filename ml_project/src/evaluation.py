from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline


def predict_proba(pipeline: Pipeline, data: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(data)[:, 1]


def evaluate_pipe(
    predicts: np.ndarray, target: pd.Series, threshold: float
) -> Dict[str, float]:
    return {
        "roc_auc": roc_auc_score(target, predicts),
        "f1_score": f1_score(target, predicts > threshold),
        "accuracy": accuracy_score(target, predicts > threshold),
    }
