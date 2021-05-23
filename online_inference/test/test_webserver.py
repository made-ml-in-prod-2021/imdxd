from unittest.mock import patch

import pytest

from src import deserialize_pipe
from webserver import app as test_app, validate_data

MODEL_PATH = "test/test_data/model.pkl"
TEST_DATA_FIRST = [
    {
        "sex": 0,
        "cp": 0,
        "fbs": 1,
        "restecg": 1,
        "exang": 2,
        "slope": "a",
        "ca": 2,
        "thal": 0,
        "age": 62.457166718365514,
        "trestbps": 135.385898507594,
        "chol": 245.9639408831817,
        "thalach": 123.89719127881307,
        "oldpeak": 1,
    },
    {
        "sex": 0,
        "cp": 0,
        "fbs": 2,
        "restecg": 1,
        "exang": 2,
        "slope": 2,
        "ca": 4,
        "thal": 3,
        "age": 49.028409686500794,
        "trestbps": 108.40035137870098,
        "chol": 202.2469211482162,
        "thalach": 178.93787774664125,
        "oldpeak": -0.5054909398188783,
    },
    {
        "sex": 2,
        "cp": 2,
        "fbs": 0,
        "restecg": 2,
        "exang": 0,
        "slope": 1,
        "ca": 3,
        "thal": 1,
        "age": 60.20525187838582,
        "trestbps": 115.69445124126652,
        "chol": 203.3837380360257,
        "thalach": 178.68649717813682,
        "oldpeak": 1.0019616918936374,
    },
]


@pytest.fixture
def client():
    with test_app.test_client() as client:
        yield client


@pytest.fixture()
def model():
    return deserialize_pipe(MODEL_PATH)


@pytest.mark.parametrize(
    "data, expected",
    [
        pytest.param(
            [
                {
                    "sex": 0,
                    "cp": 0,
                    "fbs": 1,
                    "restecg": 1,
                    "exang": 2,
                    "slope": 2,
                    "ca": 2,
                    "thal": 0,
                    "trestbps": 135.385898507594,
                    "chol": 245.9639408831817,
                    "thalach": 123.89719127881307,
                    "oldpeak": 1.5565520604484364,
                }
            ],
            False,
        ),
        pytest.param(
            [
                {
                    "sex": 0,
                    "cp": 0,
                    "fbs": 1,
                    "restecg": 1,
                    "exang": 2,
                    "slope": "a",
                    "ca": 2,
                    "thal": 0,
                    "age": 62.457166718365514,
                    "trestbps": 135.385898507594,
                    "chol": 245.9639408831817,
                    "thalach": 123.89719127881307,
                    "oldpeak": 1,
                },
                {
                    "sex": 0,
                    "cp": 0,
                    "fbs": 1.5,
                    "restecg": 1,
                    "exang": 2,
                    "slope": "a",
                    "ca": 2,
                    "thal": 0,
                    "age": 62.457166718365514,
                    "trestbps": 135.385898507594,
                    "chol": 245.9639408831817,
                    "thalach": 123.89719127881307,
                    "oldpeak": 1,
                },
            ],
            False,
        ),
        pytest.param([], False),
        pytest.param(TEST_DATA_FIRST, True),
    ],
)
def test_validation(data, expected, model):
    assert validate_data(data, model) == expected


@patch("webserver.validate_data")
def test_predict_bad_data(valid_mock, client):
    expected = "Wrong data types"
    valid_mock.return_value = False
    result = client.post("/predict", json=[])
    assert (
        400 == result.status_code
    ), f"Wrong status_code expected: 400, got: {result.status_code}"
    assert (
        expected == result.data.decode()
    ), f"Wrong message expected: {expected}, got: {result.data.decode()}"


@patch("webserver.validate_data")
def test_predict_good_data(valid_mock, client, model):
    client.application.config["model"] = model
    expected_answer_len = len(TEST_DATA_FIRST)
    expected_min_value = 0
    expected_max_value = 1
    valid_mock.return_value = True
    result = client.post("/predict", json=TEST_DATA_FIRST)
    result_data = result.json
    assert (
        200 == result.status_code
    ), f"Wrong status_code expected: 200, got: {result.status_code}"
    assert (
        expected_answer_len == len(result_data)
    ), f"Wrong len of answer expected: {expected_answer_len}, got {len(result_data)}"
    assert (
            expected_min_value == min(result_data)
    ), f"Wrong min of answer expected: {expected_min_value}, got {min(result_data)}"
    assert (
            expected_max_value == max(result_data)
    ), f"Wrong max of answer expected: {expected_max_value}, got {max(result_data)}"
    assert all([isinstance(var, int) for var in result_data]), "Wrong type of answer"
