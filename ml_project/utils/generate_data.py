import argparse
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw_data")


def generate_reals(df: pd.DataFrame):
    df["age"] = np.random.normal(54.3, 9.082, df.shape[0]).astype(np.int32)
    df["trestbps"] = np.random.normal(54.3, 9.082, df.shape[0]).astype(np.int32)
    df["chol"] = np.random.normal(54.3, 9.082, df.shape[0]).astype(np.int32)
    df["thalach"] = np.random.normal(54.3, 9.082, df.shape[0]).astype(np.int32)
    df["oldpeak"] = np.round(np.random.normal(54.3, 9.082, df.shape[0]), 3)
    df["target"] = (df["oldpeak"] == 0) & (df["thalach"] > 150)
    df["target"] = df["target"].astype(np.uint8)


def generate_categories(df: pd.DataFrame):
    df["sex"] = np.random.randint(0, 2, df.shape[0])
    df["cp"] = np.random.randint(0, 4, df.shape[0])
    df["fbs"] = np.random.randint(0, 2, df.shape[0])
    df["restecg"] = np.random.randint(0, 3, df.shape[0])
    df["exang"] = np.random.randint(0, 2, df.shape[0])
    df["ca"] = np.random.randint(0, 5, df.shape[0])
    df["thal"] = np.random.randint(0, 4, df.shape[0])


def main():
    parser = argparse.ArgumentParser(prog="simulation data generation")
    parser.add_argument("--size", type=int, help="size of new data", required=True)
    parser.add_argument(
        "--output", type=str, help="filename for storing data", required=True
    )
    arguments = parser.parse_args()
    save_path = DATA_DIR / arguments.output
    df = pd.DataFrame({"target": np.arange(arguments.size)})
    generate_categories(df)
    generate_reals(df)
    df.to_csv(save_path)


if __name__ == "__main__":
    main()
