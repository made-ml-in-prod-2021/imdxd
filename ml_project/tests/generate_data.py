import argparse

import numpy as np
import pandas as pd

from src.constants import CATEGORICAL_COLUMNS, TEST_DATA_DIR, REAL_COLUMNS, LABEL_COL


def main():
    """
    Parsing args and generating new data
    :return: Nothing
    """
    parser = argparse.ArgumentParser(prog="simulation data generation")
    parser.add_argument("--size", type=int, help="size of new data", required=True)
    parser.add_argument(
        "--output", type=str, help="filename for storing data", required=True
    )
    arguments = parser.parse_args()
    save_path = TEST_DATA_DIR / arguments.output
    df = pd.DataFrame({LABEL_COL: np.random.randint(0, 2, arguments.size)})

    for distr in REAL_COLUMNS.values():
        df[distr.name] = np.random.normal(distr.mean, distr.std, arguments.size)

    for distr in CATEGORICAL_COLUMNS.values():
        df[distr.name] = np.random.randint(0, distr.nunique + 1, arguments.size)

    df.to_csv(save_path)


if __name__ == "__main__":
    main()
