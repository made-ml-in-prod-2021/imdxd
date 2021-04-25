import pytest
import pandas as pd
from src.encoders import MeanEncoder

TEST_DATA = pd.DataFrame({"cat1": [1, 1, 0, 0], "cat2": [1, 0, 0, 1]})

TEST_TARGET = pd.Series([1, 1, 1, 0])


def test_mean_encoder_init():
    encoder = MeanEncoder(alpha=10)
    assert encoder.alpha == 10, "Wrong alpha parameter"
    assert encoder.global_mean is None, "Wrong initialization for global mean"
    assert len(encoder.cols_values) == 0, "Wrong initialization for cols values mean"


@pytest.mark.parametrize(
    "alpha, expected",
    [
        pytest.param(0, {"cat1": {1: 1, 0: 0.5}, "cat2": {1: 0.5, 0: 1}}),
        pytest.param(
            2,
            {
                "cat1": {1: (0.75 * 2 + 2) / (2 + 2), 0: (0.75 * 2 + 1) / (2 + 2)},
                "cat2": {1: (0.75 * 2 + 1) / (2 + 2), 0: (0.75 * 2 + 2) / (2 + 2)},
            },
        ),
    ],
)
def test_mean_encoder_fit(alpha, expected):
    expected_mean = TEST_TARGET.mean()
    encoder = MeanEncoder(alpha=alpha)
    encoder = encoder.fit(TEST_DATA, TEST_TARGET)
    assert (
        encoder.cols_values == expected
    ), f"Wrong means, expected: {expected}, got: {encoder.cols_values}"
    assert (
        encoder.global_mean == expected_mean
    ), f"Wrong global mean, expected: {expected_mean}, got: {encoder.global_mean}"


@pytest.mark.parametrize(
    "alpha, expected",
    [
        pytest.param(
            0, pd.DataFrame({"cat1": [1, 1, 0.5, 0.5], "cat2": [0.5, 1, 1, 0.5]})
        ),
        pytest.param(
            2,
            pd.DataFrame(
                {
                    "cat1": [3.5 / 4, 3.5 / 4, 2.5 / 4, 2.5 / 4],
                    "cat2": [2.5 / 4, 3.5 / 4, 3.5 / 4, 2.5 / 4],
                }
            ),
        ),
    ],
)
def test_mean_encoder_transform(alpha, expected):
    encoder = MeanEncoder(alpha=alpha)
    encoder = encoder.fit(TEST_DATA, TEST_TARGET)
    assert (
        encoder.transform(TEST_DATA).values == expected.values
    ).all(), f"Wrong means, expected: {expected}, got: {encoder.cols_values}"
