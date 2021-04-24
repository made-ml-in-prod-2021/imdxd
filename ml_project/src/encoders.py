from typing import NoReturn

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, alpha: int = 20) -> NoReturn:
        super(MeanEncoder, self).__init__()
        self.alpha = alpha
        self.cols_values = dict()
        self.global_mean = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> NoReturn:
        self.global_mean = y.mean()
        for col in x.columns:
            col_dict = (
                y.groupby(x[col]).sum()
                + self.global_mean * self.alpha / (x.shape[0] + self.alpha)
            ).to_dict()
            self.cols_values[col] = col_dict
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        for col in x.columns:
            x[col] = x[col].map(self.cols_values[col]).fillna(self.global_mean)
        return x
