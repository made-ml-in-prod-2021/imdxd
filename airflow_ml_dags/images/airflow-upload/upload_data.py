import os
from dataclasses import dataclass
from typing import Dict

import click
import numpy as np
import pandas as pd


@dataclass()
class RealValueCol:

    name: str
    mean: float
    std: float


@dataclass()
class CategoricalCol:

    name: str
    nunique: int


DATA_FILE_NAME = "data.csv"
TARGET_FILE_NAME = "target.csv"
LABEL_COL = "target"


CATEGORICAL_COLUMNS = {
    "sex": CategoricalCol("sex", 2),
    "cp": CategoricalCol("cp", 4),
    "fbs": CategoricalCol("fbs", 2),
    "restecg": CategoricalCol("restecg", 3),
    "exang": CategoricalCol("exang", 2),
    "slope": CategoricalCol("slope", 3),
    "ca": CategoricalCol("ca", 5),
    "thal": CategoricalCol("thal", 4),
}


REAL_COLUMNS = {
    "age": RealValueCol("age", 54.366, 9.082),
    "trestbps": RealValueCol("trestbps", 131.624, 17.538),
    "chol": RealValueCol("chol", 246.264, 51.831),
    "thalach": RealValueCol("thalach", 149.647, 22.905),
    "oldpeak": RealValueCol("oldpeak", 1.04, 1.61),
}


def generate_real(size: int) -> Dict[str, np.ndarray]:
    """
    Real values generation
    :param size: size of data
    :return: dictionary with values for columns
    """
    real_values = {}
    for name, distr in REAL_COLUMNS.items():
        real_values[name] = np.random.normal(distr.mean, distr.std, size)
    return real_values


def generate_category(size: int) -> Dict[str, np.ndarray]:
    """
    Categorical values generation
    :param size: size of data
    :return: dictionary with values for columns
    """
    categorical_values = {}
    for name, distr in CATEGORICAL_COLUMNS.items():
        categorical_values[name] = np.random.randint(0, distr.nunique + 1, size)
    return categorical_values


def generate_data(size: int) -> pd.DataFrame:
    """
    Data generation function
    :param size: size of data
    :return: dataframe with synthetic data
    """
    data = generate_real(size)
    data.update(generate_category(size))
    data = pd.DataFrame(data)
    zero_idx = np.random.choice(np.arange(data.shape[0]), size // 4, replace=False)
    data.loc[zero_idx, "oldpeak"] = 0
    data[LABEL_COL] = ((data["oldpeak"] == 0) | (data["trestbps"] > 150)).astype(
        np.uint8
    )
    return data


@click.command("upload")
@click.argument("output_dir")
def upload(output_dir: str):
    """
    Simulation of uploading data
    :param output_dir: directory to save data
    """
    df = generate_data(500)
    os.makedirs(output_dir, exist_ok=True)
    output_data = os.path.join(output_dir, DATA_FILE_NAME)
    output_target = os.path.join(output_dir, TARGET_FILE_NAME)
    df.drop(columns=LABEL_COL).to_csv(output_data, index=False)
    df[LABEL_COL].to_csv(output_target, index=False)


if __name__ == "__main__":
    upload()
