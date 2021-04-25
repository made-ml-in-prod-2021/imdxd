from typing import NoReturn

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MeanEncoder(BaseEstimator, TransformerMixin):
    """
    Mean encoder with smoothing regularization
    https://necromuralist.github.io/kaggle-competitions/posts/mean-encoding/
    """

    def __init__(self, alpha: int = 20) -> NoReturn:
        super(MeanEncoder, self).__init__()
        self.alpha = alpha
        self.cols_values = dict()
        self.global_mean = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "MeanEncoder":
        """
        Computing means and storing in class attributes
        :param x: data to fit
        :param y: labels of data
        :return: fitted exemplar of class
        """
        self.global_mean = y.mean()
        for col in x.columns:
            target_stat = y.groupby(x[col]).agg(["sum", "count"])
            col_dict = (
                (target_stat["sum"] + self.global_mean * self.alpha)
                / (target_stat["count"] + self.alpha)
            ).to_dict()
            self.cols_values[col] = col_dict
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Transforming data categories to computed means
        :param x: data to transform
        :return: transformed data
        """
        for col in x.columns:
            x[col] = x[col].map(self.cols_values[col]).fillna(self.global_mean)
        return x
