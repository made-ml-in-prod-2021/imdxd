import os

import click
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2
LABEL = "target"
DATA_FILE_NAME = "data.csv"
TARGET_FILE_NAME = "target.csv"
TRAIN_NAME = "train.csv"
VALID_NAME = "valid.csv"


@click.command("split_train_val_data")
@click.option("--input_dir")
@click.option("--output_dir")
def split_train_val_data(input_dir: str, output_dir: str):
    """
    Splitting data for training and test
    :param input_dir: directory with uploaded data
    :param output_dir: directory to save data
    """
    input_path_data = os.path.join(input_dir, DATA_FILE_NAME)
    input_path_target = os.path.join(input_dir, TARGET_FILE_NAME)
    data = pd.read_csv(input_path_data)
    target = pd.read_csv(input_path_target)
    data[LABEL] = target[LABEL]
    train_data, val_data = train_test_split(
        data, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=data[LABEL],
    )
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, TRAIN_NAME)
    valid_path = os.path.join(output_dir, VALID_NAME)
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(valid_path, index=False)


if __name__ == "__main__":
    split_train_val_data()
