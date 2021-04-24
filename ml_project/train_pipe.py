import argparse
import json
import logging

import pandas as pd

from src import (
    add_zero_features,
    evaluate_pipe,
    get_model,
    serialize_model,
    predict_proba,
    split_train_val_data,
)
from src.configs import TrainingPipelineParams, read_training_pipeline_params

logger = logging.getLogger(__name__)
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


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    logger.info(f"start train pipeline with params {training_pipeline_params}")

    data = pd.read_csv(training_pipeline_params.raw_data)

    logger.info(f"data.shape is {data.shape}")

    data = add_zero_features(data, training_pipeline_params.feature_params)

    train_df, val_df = split_train_val_data(
        data,
        training_pipeline_params.splitting_params,
        training_pipeline_params.label,
        training_pipeline_params.random_state,
    )
    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")

    model = get_model(
        training_pipeline_params.train_params,
        training_pipeline_params.feature_params,
        training_pipeline_params.random_state,
    )

    model = model.fit(train_df, train_df[training_pipeline_params.label])

    predicts = predict_proba(model, val_df)
    metrics = evaluate_pipe(
        predicts,
        val_df[training_pipeline_params.label],
        threshold=training_pipeline_params.threshold,
    )

    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics is {metrics}")

    serialize_model(model, training_pipeline_params.model_path)


def main():
    parser = argparse.ArgumentParser(prog="script for training and evaluating model")
    parser.add_argument(
        "--config", dest="config_path", help="path to pipeline config", required=True
    )
    args = parser.parse_args()
    params = read_training_pipeline_params(args.config_path)
    train_pipeline(params)


if __name__ == "__main__":
    main()
