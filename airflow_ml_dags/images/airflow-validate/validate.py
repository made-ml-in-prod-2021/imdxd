import json
import os
import pickle

import click
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

LABEL_COL = "target"
VALID_NAME = "valid.csv"
MODEL_NAME = "model.pkl"
METRIC_NAME = "metric.json"


@click.command("validate")
@click.option("--input_dir")
@click.option("--metric_dir")
@click.option("--model_dir")
def validate(input_dir: str, metric_dir: str, model_dir: str):
    model_path = os.path.join(model_dir, MODEL_NAME)
    metric_path = os.path.join(metric_dir, METRIC_NAME)
    input_data_path = os.path.join(input_dir, VALID_NAME)
    os.makedirs(metric_dir, exist_ok=True)

    with open(model_path, "rb") as fio:
        model = pickle.load(fio)
    data = pd.read_csv(input_data_path)
    predictions = model.predict(data.drop(columns=LABEL_COL))

    metrics = {
        "accuracy": accuracy_score(data[LABEL_COL], predictions),
        "f1_score": f1_score(data[LABEL_COL], predictions),
        "precision": precision_score(data[LABEL_COL], predictions),
        "recall": recall_score(data[LABEL_COL], predictions),
    }
    with open(metric_path, "w") as fio:
        json.dump(metrics, fio)


if __name__ == "__main__":
    validate()
