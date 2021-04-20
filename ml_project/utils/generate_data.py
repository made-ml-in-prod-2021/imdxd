import argparse

import numpy as np
import pandas as pd

CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

REAL_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def generate_reals(df: pd.DataFrame):
    df["age"] = np.random.normal(54.3, 9.082, df.shape[0])
    df["trestbps"] = np.random.normal(54.3, 9.082, df.shape[0])
    df["chol"] = np.random.normal(54.3, 9.082, df.shape[0])
    df["thalach"] = np.random.normal(54.3, 9.082, df.shape[0])
    df["oldpeak"] = np.random.normal(54.3, 9.082, df.shape[0])






if __name__ == '__main__':
    main()