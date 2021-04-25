import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, NoReturn

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
class SerilizedModel:

    pipeline: Pipeline
    zero_cols: List[str]


def get_real_feature_pipe(params: TrainingParams) -> Pipeline:
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=params.imput_strategy)),
            ("scaler", StandardScaler()),
        ]
    )
    return pipe


def get_cat_feature_pipe(params: TrainingParams) -> Pipeline:
    pipe = Pipeline(
        [
            ("encoder", MeanEncoder(params.mean_alpha)),
            ("imputer", SimpleImputer(fill_value=-1)),
        ]
    )
    return pipe


def get_model(
    train_params: TrainingParams, feature_params: FeatureParams, random_state: int
) -> Pipeline:
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression(random_state=random_state)
    elif train_params.model_type == "GradientBoostingClassifier":
        model = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
    else:
        raise NotImplementedError()

    cat_cols = feature_params.cat_cols
    cat_cols.extend([f"zero_{col}" for col in feature_params.zero_cols])

    logger.debug("Cat cols are: %s" % ", ".join(cat_cols))
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
                            ("cat", get_cat_feature_pipe(train_params), cat_cols,),
                        ]
                    ),
                ),
                ("model", model),
            ]
        )
    )

    return pipeline


def serialize_pipe(model: Pipeline, output: Path, params: FeatureParams) -> NoReturn:
    logger.debug("Serialized model to: %s" % output)
    with open(output, "wb") as f:
        pickle.dump(SerilizedModel(pipeline=model, zero_cols=params.zero_cols), f)


def deserialize_pipe(input_: Path) -> SerilizedModel:
    logger.debug("Deserialized model from: %s" % input_)
    with open(input_, "rb") as f:
        return pickle.load(f)
