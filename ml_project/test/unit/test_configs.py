import pytest
from marshmallow.exceptions import ValidationError

from src.configs import read_evaluation_pipeline_params

TEST_WRONG_EVAL_CONFIG_PATH = "test/test_data/wrong_evaluation.yaml"
TEST_RIGHT_DEFAULT_EVAL_CONFIG_PATH = "test/test_data/right_evaluation_def.yaml"
TEST_RIGHT_EVAL_CONFIG_PATH = "test/test_data/right_evaluation.yaml"


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


def test_exception_eval_config():
    with pytest.raises(ValidationError):
        read_evaluation_pipeline_params(TEST_WRONG_EVAL_CONFIG_PATH)
