import pytest
from marshmallow.exceptions import ValidationError

from src.configs import read_evaluation_pipeline_params, read_training_pipeline_params

TEST_WRONG_EVAL_CONFIG_PATH = "test/test_data/wrong_evaluation.yaml"
TEST_RIGHT_DEFAULT_EVAL_CONFIG_PATH = "test/test_data/right_evaluation_def.yaml"
TEST_RIGHT_EVAL_CONFIG_PATH = "test/test_data/right_evaluation.yaml"

TEST_WRONG_TRAIN_CONFIG_PATH = "test/test_data/wrong_train.yaml"
TEST_RIGHT_DEFAULT_TRAIN_CONFIG_PATH = "test/test_data/right_train_def.yaml"
TEST_RIGHT_TRAIN_CONFIG_PATH = "test/test_data/right_train.yaml"


@pytest.mark.parametrize(
    "config_path, expected",
    [
        pytest.param(
            TEST_RIGHT_EVAL_CONFIG_PATH,
            {
                "model": "rf",
                "raw_data": "heart.csv",
                "proceed_data": "rf.csv",
                "threshold": 0.1,
            },
        ),
        pytest.param(
            TEST_RIGHT_DEFAULT_EVAL_CONFIG_PATH,
            {
                "model": "rf",
                "raw_data": "heart.csv",
                "proceed_data": "rf.csv",
                "threshold": 0.5,
            },
        ),
    ],
)
def test_read_eval_config(config_path, expected):
    config = read_evaluation_pipeline_params(config_path)
    data_params = {
        "model": config.model,
        "raw_data": config.raw_data,
        "proceed_data": config.proceed_data,
        "threshold": config.threshold,
    }
    assert set(data_params.keys()) == set(
        expected.keys()
    ), f"Wrong params, expected: {expected}, got: {data_params}"
    for key in data_params:
        assert (
            data_params[key] == expected[key]
        ), f"Wrong param {key} value, expected: {expected[key]}, got: {data_params[key]}"


@pytest.mark.parametrize(
    "config_path, expected",
    [
        pytest.param(
            TEST_RIGHT_TRAIN_CONFIG_PATH,
            {
                "raw_data": "heart.csv",
                "experiment_name": "gbm",
                "random_state": 42,
                "label": "target",
                "splitting_params": {"val_size": 0.1, "stratify": True},
                "train_params": {
                    "model_type": "GradientBoostingClassifier",
                    "mean_alpha": 10,
                    "imput_strategy": "mean",
                },
                "feature_params": {
                    "cat_cols": [
                        "sex",
                        "cp",
                        "fbs",
                        "restecg",
                        "exang",
                        "slope",
                        "ca",
                        "thal",
                    ],
                    "real_cols": ["age", "trestbps", "chol", "thalach", "oldpeak"],
                    "zero_cols": ["oldpeak"],
                },
            },
        ),
        pytest.param(
            TEST_RIGHT_DEFAULT_TRAIN_CONFIG_PATH,
            {
                "raw_data": "heart.csv",
                "experiment_name": "gbm",
                "random_state": 42,
                "label": "target",
                "splitting_params": {"val_size": 0.2, "stratify": False},
                "train_params": {
                    "model_type": "GradientBoostingClassifier",
                    "mean_alpha": 10,
                    "imput_strategy": "mean",
                },
                "feature_params": {
                    "cat_cols": [
                        "sex",
                        "cp",
                        "fbs",
                        "restecg",
                        "exang",
                        "slope",
                        "ca",
                        "thal",
                    ],
                    "real_cols": ["age", "trestbps", "chol", "thalach", "oldpeak"],
                    "zero_cols": ["oldpeak"],
                },
            },
        ),
    ],
)
def test_read_train_config(config_path, expected):
    config = read_training_pipeline_params(config_path)
    train_params = {
        "raw_data": config.raw_data,
        "experiment_name": config.experiment_name,
        "random_state": config.random_state,
        "label": config.label,
        "splitting_params": {
            "val_size": config.splitting_params.val_size,
            "stratify": config.splitting_params.stratify,
        },
        "train_params": {
            "model_type": config.train_params.model_type,
            "mean_alpha": config.train_params.mean_alpha,
            "imput_strategy": config.train_params.imput_strategy,
        },
        "feature_params": {
            "cat_cols": config.feature_params.cat_cols,
            "real_cols": config.feature_params.real_cols,
            "zero_cols": config.feature_params.zero_cols,
        },
    }
    assert set(train_params.keys()) == set(
        expected.keys()
    ), f"Wrong params, expected: {expected}, got: {train_params}"
    for key in train_params:
        assert (
            train_params[key] == expected[key]
        ), f"Wrong param {key} value, expected: {expected[key]}, got: {train_params[key]}"


def test_exception_eval_config():
    with pytest.raises(ValidationError):
        read_evaluation_pipeline_params(TEST_WRONG_EVAL_CONFIG_PATH)


def test_exception_train_config():
    with pytest.raises(ValidationError):
        read_training_pipeline_params(TEST_WRONG_TRAIN_CONFIG_PATH)
