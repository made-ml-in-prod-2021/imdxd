import json
import logging
import os

import hydra
import pandas as pd
from omegaconf import DictConfig

from src import (
    add_zero_features,
    compute_metrics,
    get_column_order,
    get_full_pipeline,
    serialize_pipe,
    predict_proba,
    split_train_val_data,
)
from src.configs import TrainingPipelineParams, TrainingPipelineParamsSchema
from src.constants import ARTIFACT_DIR, DATA_DIR

logger = logging.getLogger("ml_project")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
file_handler = logging.FileHandler("logs/train.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
stdout_handler = logging.StreamHandler()
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)


def train_pipeline(pipe_params: TrainingPipelineParams):
    """
    Training model, computing metrics and storing model
    :param pipe_params: params for training
    :return: Nothing
    """
    logger.info(f"start train pipeline with params {pipe_params}")

    data = pd.read_csv(DATA_DIR / pipe_params.raw_data)

    experiment_path = ARTIFACT_DIR / pipe_params.experiment_name
    experiment_path.mkdir(parents=True, exist_ok=False)

    logger.info(f"data.shape is {data.shape}")

    data = add_zero_features(data, pipe_params.feature_params.zero_cols)

    train_df, val_df = split_train_val_data(
        data, pipe_params.splitting_params, pipe_params.label, pipe_params.random_state,
    )
    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")

    train_df.to_csv(experiment_path / "train.csv", index=False)
    val_df.to_csv(experiment_path / "val.csv", index=False)

    model = get_full_pipeline(
        pipe_params.train_params, pipe_params.feature_params, pipe_params.random_state,
    )

    logger.info("Start training model")

    column_order = get_column_order(pipe_params.feature_params)

    model = model.fit(train_df[column_order], train_df[pipe_params.label])

    logger.info("Model training finished")
    logger.info("Start prediction")

    predicts = predict_proba(model, val_df[column_order])
    metrics = compute_metrics(
        predicts, val_df[pipe_params.label], threshold=pipe_params.threshold,
    )

    logger.info("Prediction finished")

    metric_path = experiment_path / "metric.json"
    model_path = experiment_path / "model.pkl"

    with open(metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics is {metrics}")

    serialize_pipe(model, model_path, pipe_params.feature_params)


@hydra.main(config_path="configs")
def main(cfg: DictConfig):
    """
    Wrapper for arguments reading and start training
    :return: Nothing
    """
    os.chdir(hydra.utils.to_absolute_path('.'))
    schema = TrainingPipelineParamsSchema()
    cfg = schema.load(cfg)
    train_pipeline(cfg)


if __name__ == "__main__":
    main()
