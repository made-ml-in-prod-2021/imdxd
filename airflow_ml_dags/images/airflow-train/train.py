import os
import pickle

import click
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

LABEL_COL = "target"
TRAIN_NAME = "train.csv"
N_ESTIMATORS = 100
MAX_DEPTH = 32
MIN_NUMBER_IN_LEAF = 3
RANDOM_STATE = 42
MODEL_NAME = "model.pkl"


@click.command("train")
@click.option("--input_dir")
@click.option("--model_dir")
def train(input_dir: str, model_dir: str):
    model_path = os.path.join(model_dir, MODEL_NAME)
    input_train_path = os.path.join(input_dir, TRAIN_NAME)
    os.makedirs(model_dir, exist_ok=True)

    config = {
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "min_samples_leaf": MIN_NUMBER_IN_LEAF,
        "random_state": RANDOM_STATE,
    }

    model = RandomForestClassifier(**config)
    data = pd.read_csv(input_train_path)
    model.fit(data.drop(columns=LABEL_COL), data[LABEL_COL])
    with open(model_path, "wb") as fio:
        pickle.dump(model, fio)


if __name__ == "__main__":
    train()
