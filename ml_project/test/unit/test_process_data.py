import pandas as pd
import pytest

from src.configs import SplittingParams
from src.process_data import add_zero_features, split_train_val_data

TEST_DATA = pd.DataFrame(
    {
        "f1": [3, 2, 2, 2, 1, 1, 1, 0, 0, 0],
        "f2": [3, 3, 3, 2, 2, 2, 1, 1, 1, 0],
        "label": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    }
)


@pytest.mark.parametrize(
    "zero_col, expected",
    [
        pytest.param(
            ["f1"],
            pd.DataFrame(
                {
                    "f1": [3, 2, 2, 2, 1, 1, 1, 0, 0, 0],
                    "f2": [3, 3, 3, 2, 2, 2, 1, 1, 1, 0],
                    "label": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    "zero_f1": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                }
            ),
        ),
        pytest.param([], TEST_DATA,),
        pytest.param(
            ["f1", "f2"],
            pd.DataFrame(
                {
                    "f1": [3, 2, 2, 2, 1, 1, 1, 0, 0, 0],
                    "f2": [3, 3, 3, 2, 2, 2, 1, 1, 1, 0],
                    "label": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    "zero_f1": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    "zero_f2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                }
            ),
        ),
    ],
)
def test_add_zero_features(zero_col, expected):
    data = add_zero_features(TEST_DATA, zero_col)
    assert set(data.columns) == set(
        expected.columns
    ), f"Wrong columns, expected {expected.columns}, got: {data.columns}"
    for col in zero_col:
        assert (
            data[f"zero_{col}"] == expected[f"zero_{col}"]
        ).all(), f"Wrong calculated features for col: {col}"


def test_split_train_stratify():
    params = SplittingParams(val_size=0.2, stratify=True)
    train, val = split_train_val_data(TEST_DATA, params, "label", 42)
    assert (
        train.shape[0] / 4 == val.shape[0]
    ), "Size of train must be 4 times higher val size"
    assert (
        train["label"].sum() == val["label"].sum() * 4
    ), "Positive target must have the same fraction"
    assert (1 - train["label"]).sum() == (
        1 - val["label"]
    ).sum() * 4, "Negative target must have the same fraction"


def test_split_train_not_stratify():
    params = SplittingParams(val_size=0.1, stratify=False)
    train, val = split_train_val_data(TEST_DATA, params, "label", 42)
    assert (
        train.shape[0] == val.shape[0] * 9
    ), "Size of train must be 9 times higher val size"


def test_split_train_to_less_stratify():
    with pytest.raises(ValueError):
        params = SplittingParams(val_size=0.1, stratify=True)
        train, val = split_train_val_data(TEST_DATA, params, "label", 42)
