from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import get_column_order, deserialize_pipe
from src.evaluation import predict_proba, compute_metrics

TEST_MODEL_PATH = Path("test/test_data/test_model.pkl")
TEST_DATA = "test/test_data/test_gen.csv"

TEST_TARGET = pd.Series([1, 1, 1, 0, 0, 0])


def test_predict_proba():

    model = deserialize_pipe(TEST_MODEL_PATH)
    data = pd.read_csv(TEST_DATA)
    col_order = get_column_order(model.feature_params)
    predict = predict_proba(model.pipeline, data[col_order])
    assert predict.shape[0] == data.shape[0], "Wrong prediction size"
    assert (predict <= 1).all(), "Wrong prediction max"
    assert (predict >= 0).all(), "Wrong prediction min"
    assert (
        np.unique(predict).shape[0] > 2
    ), "Prediction of labels instead of probability"


@pytest.mark.parametrize(
    "prediction, threshold, expected",
    [
        pytest.param(
            np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
            0.55,
            {"roc_auc": 1, "accuracy": 5 / 6, "f1_score": 1.5 / 1.75},
        ),
        pytest.param(
            np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]),
            0.5,
            {"roc_auc": 0, "accuracy": 0, "f1_score": 0},
        ),
    ],
)
def test_compute_metrics(prediction, threshold, expected):
    metrics = compute_metrics(prediction, TEST_TARGET, threshold)
    assert metrics == expected, f"Wrong metrics, expected: {expected}, got: {metrics}"
