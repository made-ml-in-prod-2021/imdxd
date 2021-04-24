import pickle
from pathlib import Path
from typing import NoReturn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.configs import FeatureParams, TrainingParams
from .encoders import MeanEncoder


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


def serialize_model(model: Pipeline, output: Path) -> NoReturn:
    with open(output, "wb") as f:
        pickle.dump(model, f)
