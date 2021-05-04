import os
import json

import pytest

from src import (
    add_zero_features,
    compute_metrics,
    get_column_order,
    get_full_pipeline,
    serialize_pipe,
    predict_proba,
    split_train_val_data,
)
from src.configs import (
    FeatureParams,
    SplittingParams,
    TrainingParams,
    TrainingPipelineParams,
)
from test import generate_data


@pytest.fixture()
def pipe_params():
    return TrainingPipelineParams(
        raw_data="",
        experiment_name="gbm",
        random_state=42,
        label="target",
        splitting_params=SplittingParams(val_size=0.1, stratify=True),
        train_params=TrainingParams(
            model_type="GradientBoostingClassifier",
            mean_alpha=10,
            imput_strategy="mean",
        ),
        feature_params=FeatureParams(
            cat_cols=["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
            real_cols=["age", "trestbps", "chol", "thalach", "oldpeak"],
            zero_cols=["oldpeak"],
        ),
    )


def test_train_pipeline(tmp_path, pipe_params):

    data = generate_data(500)

    experiment_path = tmp_path / pipe_params.experiment_name
    experiment_path.mkdir(parents=True)

    data = add_zero_features(data, pipe_params.feature_params.zero_cols)

    train_df, val_df = split_train_val_data(
        data, pipe_params.splitting_params, pipe_params.label, pipe_params.random_state,
    )

    train_df.to_csv(experiment_path / "train.csv", index=False)
    val_df.to_csv(experiment_path / "val.csv", index=False)

    model = get_full_pipeline(
        pipe_params.train_params, pipe_params.feature_params, pipe_params.random_state,
    )

    column_order = get_column_order(pipe_params.feature_params)

    model = model.fit(train_df[column_order], train_df[pipe_params.label])

    predicts = predict_proba(model, val_df[column_order])
    metrics = compute_metrics(
        predicts, val_df[pipe_params.label], threshold=pipe_params.threshold,
    )

    metric_path = experiment_path / "metric.json"
    model_path = experiment_path / "model.pkl"

    with open(metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)

    serialize_pipe(model, model_path, pipe_params.feature_params)

    assert os.path.isfile(experiment_path / "train.csv")
    assert os.path.isfile(experiment_path / "val.csv")
    assert os.path.isfile(experiment_path / "metric.json")
    assert os.path.isfile(experiment_path / "model.pkl")
    assert metrics["accuracy"] > 0.7
