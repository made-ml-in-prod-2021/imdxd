from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.configs import FeatureParams, SplittingParams


def add_zero_features(data: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:

    for col in params.zero_cols:
        data[f"zero_{col}"] = (data[col] == 0).astype(np.uint8)
    return data


def split_train_val_data(
    data: pd.DataFrame, params: SplittingParams, label: str, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if params.stratify:
        train_data, val_data = train_test_split(
            data,
            test_size=params.val_size,
            random_state=random_state,
            stratify=data[label],
        )
    else:
        train_data, val_data = train_test_split(
            data, test_size=params.val_size, random_state=random_state,
        )
    return train_data, val_data
