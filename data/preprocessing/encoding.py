import pandas as pd
from sklearn.base import TransformerMixin


class BinaryEncoder(TransformerMixin):
    def __init__(self, col: str, normal_label: str):
        self.col = col
        self.normal_label = normal_label
        self.before_shape = None
        self.after_shape = None

    def fit(self, df: pd.DataFrame, y: pd.DataFrame = None):
        self.before_shape = df.shape
        return self

    def transform(self, df: pd.DataFrame, y: pd.DataFrame = None):
        # Convert column to binary labels
        df.loc[df[self.col].isin(self.normal_label), self.col] = 0
        df.loc[df[self.col] != 0, self.col] = 1
        self.after_shape = df.shape
        return df
