import logging
import pickle
from pathlib import Path
from typing import List, NoReturn

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.configs import FeatureParams, TrainingParams
from .encoders import MeanEncoder

logger = logging.getLogger("ml_project")


@dataclass()
class SerializedModel:
    """
    Data class for storing fitted pipeline
    """

    pipeline: Pipeline
    feature_params: FeatureParams


def get_real_feature_pipe(params: TrainingParams) -> Pipeline:
    """
    Get pipeline for real value columns
    :param params: params for training
    :return: pipeline for real data processing
    """
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=params.imput_strategy)),
            ("scaler", StandardScaler()),
        ]
    )
    return pipe


def get_cat_feature_pipe(params: TrainingParams) -> Pipeline:
    """
    Get pipeline for categorical columns
    :param params: params for training
    :return: pipeline for categorical data processing
    """
    pipe = Pipeline(
        [
            ("encoder", MeanEncoder(params.mean_alpha)),
            ("imputer", SimpleImputer(fill_value=-1)),
        ]
    )
    return pipe


def get_full_pipeline(
    train_params: TrainingParams, feature_params: FeatureParams, random_state: int
) -> Pipeline:
    """
    Get pipeline for full data processing
    :param train_params: params for training
    :param feature_params: params for columns definitions
    :param random_state: random state for reproducibility
    :return: full pipeline for data processing
    """
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression(random_state=random_state)
    elif train_params.model_type == "GradientBoostingClassifier":
        model = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
    else:
        raise NotImplementedError()

    logger.debug("Cat cols are: %s" % ", ".join(feature_params.cat_cols))
    logger.debug("Real cols are: %s" % ", ".join(feature_params.real_cols))

    pipeline = Pipeline(
        (
            [
                (
                    "preprocess",
                    ColumnTransformer(
                        [
                            (
                                "real",
                                get_real_feature_pipe(train_params),
                                feature_params.real_cols,
                            ),
                            (
                                "cat",
                                get_cat_feature_pipe(train_params),
                                feature_params.cat_cols,
                            ),
                        ]
                    ),
                ),
                ("model", model),
            ]
        )
    )

    return pipeline


def get_column_order(params: FeatureParams) -> List[str]:
    """
    Return order of data columns
    :param params: parameters of columns
    :return: ordered columns
    """
    renamed_zero_cols = [f"zero_{col}" for col in params.zero_cols]
    return params.real_cols + params.cat_cols + renamed_zero_cols


def serialize_pipe(model: Pipeline, output: Path, params: FeatureParams) -> NoReturn:
    """
    Saving pipeline
    :param model: full model pipeline
    :param output: path to save
    :param params: columns definitions
    :return: Nothing
    """
    logger.debug("Serialized model to: %s" % output)
    with open(output, "wb") as f:
        pickle.dump(SerializedModel(pipeline=model, feature_params=params), f)


def deserialize_pipe(input_: Path) -> SerializedModel:
    """
    Loading pipeline
    :param input_: path to load
    :return: Serialized pipeline class with fitted model and columns definitions
    """
    logger.debug("Deserialized model from: %s" % input_)
    with open(input_, "rb") as f:
        return pickle.load(f)
