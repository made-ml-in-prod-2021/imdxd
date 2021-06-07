import os
import pickle

import click
import pandas as pd


LABEL_COL = "target"
DATA_NAME = "data.csv"
MODEL_NAME = "model.pkl"
PREDICTION_NAME = "prediction.csv"


@click.command("predict")
@click.option("--input_dir")
@click.option("--model_dir")
@click.option("--prediction_dir")
def predict(input_dir: str, model_dir: str, prediction_dir: str):
    model_path = os.path.join(model_dir, MODEL_NAME)
    input_data_path = os.path.join(input_dir, DATA_NAME)
    prediction_path = os.path.join(prediction_dir, PREDICTION_NAME)
    os.makedirs(prediction_dir, exist_ok=True)

    with open(model_path, "rb") as fio:
        model = pickle.load(fio)

    data = pd.read_csv(input_data_path)
    predictions = model.predict(data)
    predictions = pd.Series(predictions)
    predictions.to_csv(prediction_path, index=False)


if __name__ == "__main__":
    predict()
