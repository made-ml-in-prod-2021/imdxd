import argparse
from typing import Dict

import numpy as np
import pandas as pd

from src.constants import CATEGORICAL_COLUMNS, TEST_DATA_DIR, REAL_COLUMNS, LABEL_COL


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
    data[LABEL_COL] = ((data["oldpeak"] == 0) | (data["trestbps"] > 150)).astype(np.uint8)
    return data


def main():
    """
    Wrapper with parsing args, generating and saving new data
    :return: Nothing
    """
    parser = argparse.ArgumentParser(prog="simulation data generation")
    parser.add_argument("--size", type=int, help="size of new data", required=True)
    parser.add_argument(
        "--output", type=str, help="filename for storing data", required=True
    )
    arguments = parser.parse_args()
    save_path = TEST_DATA_DIR / arguments.output
    df = generate_data(arguments.size)
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
