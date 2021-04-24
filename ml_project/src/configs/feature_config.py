from typing import List

from dataclasses import dataclass, field


@dataclass()
class FeatureParams:

    real_cols: List[str] = field(
        default_factory=["age", "trestbps", "chol", "thalach", "oldpeak"]
    )
    cat_cols: List[str] = field(
        default_factory=["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    )
    zero_cols: List[str] = field(default_factory=[])
