import os

import click
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


LABEL_COL = "target"
TRAIN_NAME = "train.csv"
N_ESTIMATORS = 100
MAX_DEPTH = 32
MIN_NUMBER_IN_LEAF = 3
RANDOM_STATE = 42
TEST_SIZE = 0.2
EXPERIMENT_NAME = "airflow"
MODEL_NAME = "model"
MLFLOW_URI = "http://localhost:5000"


@click.command("train")
@click.option("--input_dir")
def train(input_dir: str):
    config = {
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "min_samples_leaf": MIN_NUMBER_IN_LEAF,
        "random_state": RANDOM_STATE,
    }
    model = RandomForestClassifier(**config)
    config["test_size"] = TEST_SIZE
    input_train_path = os.path.join(input_dir, TRAIN_NAME)
    data = pd.read_csv(input_train_path)
    model.fit(data.drop(columns=LABEL_COL), data[LABEL_COL])

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    for k, v in config.items():
        mlflow.log_param(k, v)
    mlflow.sklearn.log_model(model, "model.pkl", registered_model_name=MODEL_NAME)


if __name__ == "__main__":
    train()
