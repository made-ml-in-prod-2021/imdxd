import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.configs import SplittingParams

logger = logging.getLogger("ml_project")


def add_zero_features(data: pd.DataFrame, zero_cols: List[str]) -> pd.DataFrame:
    """
    Add features col == 0 for given columns
    :param data: data to process
    :param zero_cols: cols to convert
    :return: data with new "equal zero" columns
    """
    logger.debug("Adding columns %s" % " ".join([f"zero_{col}" for col in zero_cols]))
    for col in zero_cols:
        data[f"zero_{col}"] = (data[col] == 0).astype(np.uint8)
    return data


def split_train_val_data(
    data: pd.DataFrame, params: SplittingParams, label: str, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splitting data for training and test
    :param data: data to split
    :param params: splitting params
    :param label: label of data
    :param random_state: random state for reproducibility
    :return: split data to train and validation parts
    """
    if params.stratify:
        logger.debug("Split data with stratification by %s" % label)
        train_data, val_data = train_test_split(
            data,
            test_size=params.val_size,
            random_state=random_state,
            stratify=data[label],
        )
    else:
        logger.debug("Split data without stratification")
        train_data, val_data = train_test_split(
            data, test_size=params.val_size, random_state=random_state,
        )
    return train_data, val_data
