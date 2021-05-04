"""
Module with constants for application
"""

from pathlib import Path

from dataclasses import dataclass


@dataclass()
class RealValueCol:

    name: str
    mean: float
    std: float


@dataclass()
class CategoricalCol:

    name: str
    nunique: int


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

IS_ZERO_COLS = ["oldpeak"]

LABEL_COL = "target"

DATA_DIR = Path("data/raw_data")

REPORT_DIR = Path("reports")

ARTIFACT_DIR = Path("experiments")

PROCEED_DIR = Path("data/proceed_data")

TEST_DATA_DIR = Path("test/test_data")
