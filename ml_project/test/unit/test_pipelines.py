import os
import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from src.configs import FeatureParams, TrainingParams
from src.constants import CATEGORICAL_COLUMNS, LABEL_COL, REAL_COLUMNS
from src.pipelines import (
    get_real_feature_pipe,
    get_cat_feature_pipe,
    get_column_order,
    get_full_pipeline,
    serialize_pipe,
    deserialize_pipe,
)
from test import generate_data

TEST_MODEL = Path("test/test_data/test_model.pkl")


@pytest.fixture()
def training_params():
    params = TrainingParams(
        model_type="GradientBoostingClassifier", mean_alpha=10, imput_strategy="mean"
    )
    return params


@pytest.fixture()
def feature_params():
    params = FeatureParams(
        cat_cols=list(CATEGORICAL_COLUMNS.keys()),
        real_cols=list(REAL_COLUMNS.keys()),
        zero_cols=[],
    )
    return params


def test_get_real_feature_pipe(training_params):
    pipe = get_real_feature_pipe(training_params)
    data = generate_data(100)
    data.loc[:10, REAL_COLUMNS] = None
    result = pipe.fit_transform(data[REAL_COLUMNS], data[LABEL_COL])
    assert not np.isnan(result).any(), "Empty value detected"
    assert np.allclose(result.mean(0), 0, rtol=1e-2), "Wrong mean"
    assert np.allclose(result.std(0), 1, rtol=1e-2), "Wrong mean"


def test_get_cat_feature_pipe(training_params):
    pipe = get_cat_feature_pipe(training_params)
    data = generate_data(100)
    data.loc[:10, CATEGORICAL_COLUMNS] = None
    result = pipe.fit_transform(data[CATEGORICAL_COLUMNS], data[LABEL_COL])
    assert not np.isnan(result).any(), "Empty value detected"
    assert result.min() >= 0, "Negative values detected"
    assert result.max() <= 1, "Not mean value for binary label detected"


def test_get_full_pipeline(training_params, feature_params):
    pipe = get_full_pipeline(training_params, feature_params, 42)
    data = generate_data(100)
    pipe.fit(data, data[LABEL_COL])
    predictions = pipe.predict(data)
    assert accuracy_score(data[LABEL_COL], predictions) > 0.9, "Bad pipeline quality"


def test_serialize_pipe(tmp_path, training_params, feature_params):
    pipe = get_full_pipeline(training_params, feature_params, 42)
    data = generate_data(100)
    pipe.fit(data, data[LABEL_COL])
    serialize_pipe(pipe, tmp_path / "model.pkl", feature_params)
    assert os.path.isfile(tmp_path / "model.pkl"), "No saved model"
    with open(tmp_path / "model.pkl", "rb") as fio:
        loaded = pickle.load(fio)
        assert loaded.feature_params == feature_params, "Wrong zero cols"
        assert loaded.pipeline.named_steps["model"].n_classes_ == 2, "Model not fitted"


def test_deserialize_pipe(feature_params):
    pipe = deserialize_pipe(TEST_MODEL)
    assert pipe.feature_params == feature_params, "Wrong feature params"
    assert pipe.pipeline.named_steps["model"].n_classes_ == 2, "Wrong model classes"


def test_get_column_order(feature_params):
    col_order = get_column_order(feature_params)
    renamed_zero_cols = [f"zero_{col}" for col in feature_params.zero_cols]
    assert (
        col_order
        == feature_params.real_cols + feature_params.cat_cols + renamed_zero_cols
    ), "Wrong column order"
