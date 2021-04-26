import os
from pathlib import Path

import numpy as np
import pytest

from src import add_zero_features, deserialize_pipe, get_column_order, predict_proba
from src.configs import EvaluationParams
from test import generate_data

TEST_MODEL = Path("test/test_data/test_model.pkl")


@pytest.fixture()
def evaluation_params():
    return EvaluationParams(
        model="", raw_data="", proceed_data="test.csv", threshold=0.5
    )


def test_train_e2e(tmp_path, evaluation_params):

    data = generate_data(100)
    initial_size = data.shape[1]
    model = deserialize_pipe(TEST_MODEL)

    data = add_zero_features(data, model.feature_params.zero_cols)

    column_order = get_column_order(model.feature_params)

    predictions = predict_proba(model.pipeline, data[column_order])

    data["Prediction"] = (predictions > evaluation_params.threshold).astype(np.uint8)

    data.to_csv(tmp_path / evaluation_params.proceed_data, index=False)
    assert os.path.exists(
        tmp_path / evaluation_params.proceed_data
    ), "Prediction didn't saved"
    assert data.shape[1] == initial_size + 1, "Wrong data size"
