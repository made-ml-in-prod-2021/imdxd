import os
from shutil import copyfile

import click
import numpy as np
import pandas as pd

DATA_FILE_NAME = "data.csv"
TARGET_FILE_NAME = "target.csv"
LABEL_COL = "target"
ZERO_COL = "oldpeak"


@click.command("preprocess_data")
@click.option("--input_dir")
@click.option("--output_dir")
def preprocess_data(input_dir: str, output_dir: str):
    """
    Add features col == 0
    :param input_dir: directory with uploaded data
    :param output_dir: directory to save data
    """
    input_data_path = os.path.join(input_dir, DATA_FILE_NAME)
    input_target_path = os.path.join(input_dir, TARGET_FILE_NAME)
    output_target_path = os.path.join(output_dir, TARGET_FILE_NAME)
    output_path = os.path.join(output_dir, DATA_FILE_NAME)
    os.makedirs(output_dir, exist_ok=True)

    data = pd.read_csv(input_data_path)
    if os.path.isfile(input_target_path):
        copyfile(input_target_path, output_target_path)
    data[f"zero_{ZERO_COL}"] = (data[ZERO_COL] == 0).astype(np.uint8)
    data.to_csv(output_path, index=False)


if __name__ == "__main__":
    preprocess_data()
