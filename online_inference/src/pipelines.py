import pickle
from typing import List

from dataclasses import dataclass
from sklearn.pipeline import Pipeline

from .configs import FeatureParams


@dataclass()
class SerializedModel:
    """
    Data class for storing fitted pipeline
    """

    pipeline: Pipeline
    feature_params: FeatureParams


def get_column_order(params: FeatureParams) -> List[str]:
    """
    Return order of data columns
    :param params: parameters of columns
    :return: ordered columns
    """
    renamed_zero_cols = [f"zero_{col}" for col in params.zero_cols]
    return params.real_cols + params.cat_cols + renamed_zero_cols


def deserialize_pipe(input_: str) -> SerializedModel:
    """
    Loading pipeline
    :param input_: path to load
    :return: Serialized pipeline class with fitted model and columns definitions
    """
    with open(input_, "rb") as f:
        return pickle.load(f)
