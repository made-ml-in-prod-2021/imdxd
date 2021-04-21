import argparse
from pathlib import Path
from typing import NoReturn

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPORT_DIR = Path("reports")

CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

REAL_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def save_pairplot(raw_data: pd.DataFrame, output_dir: Path) -> NoReturn:
    sns.pairplot(raw_data[REAL_COLUMNS])
    plt.savefig(output_dir / "pairplot.png")


def save_heatmap(raw_data: pd.DataFrame, output_dir: Path) -> NoReturn:
    plt.figure(figsize=(10, 9))
    sns.heatmap(raw_data[REAL_COLUMNS].corr(), annot=True, linewidths=0.3)
    plt.savefig(output_dir / "heatmap.png")


def save_real_graphs(raw_data: pd.DataFrame, output_dir: Path) -> NoReturn:
    fig, axs = plt.subplots(3, 2)
    fig.set_figheight(10)
    fig.set_figwidth(16)
    for i, col in enumerate(REAL_COLUMNS):
        sns.boxplot(
            x=raw_data[col], y=raw_data["target"], ax=axs.ravel()[i], orient="h"
        )
        axs.ravel()[i].grid(True)
    plt.savefig(output_dir / "realplots.png")


def save_categorical_graphs(raw_data: pd.DataFrame, output_dir: Path) -> NoReturn:
    fig, axs = plt.subplots(3, 3)
    fig.set_figheight(14)
    fig.set_figwidth(16)
    for i, col in enumerate(CATEGORICAL_COLUMNS):
        sns.countplot(x=raw_data[col], hue=raw_data["target"], ax=axs.ravel()[i])
        axs.ravel()[i].grid(True)
    sns.countplot(x=raw_data["target"], ax=axs.ravel()[-1])
    axs.ravel()[-1].grid(True)
    plt.savefig(output_dir / "catplots.png")


def save_data_stats(raw_data: pd.DataFrame, output_dir: Path) -> NoReturn:
    descr = raw_data.describe()
    descr.loc["nunique", :] = raw_data.nunique()
    descr.loc["dtype", :] = raw_data.dtypes
    descr = descr.round(3)
    descr.to_csv(output_dir / "stats.csv")


def main():
    parser = argparse.ArgumentParser(prog="data reports generation")
    parser.add_argument(
        "--input", type=str, help="path to dataset for reports", required=True
    )
    parser.add_argument(
        "--output", type=str, help="reports folder output", required=True
    )
    arguments = parser.parse_args()
    output_dir = REPORT_DIR / arguments.output
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(arguments.input)
    save_data_stats(df, output_dir)
    save_categorical_graphs(df, output_dir)
    save_real_graphs(df, output_dir)
    save_heatmap(df, output_dir)
    save_pairplot(df, output_dir)


if __name__ == "__main__":
    main()
