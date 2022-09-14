import pandas as pd
from sklearn.base import TransformerMixin


class RenameColumn(TransformerMixin):
    def __init__(self, col: str, name: str):
        self.col = col
        self.name = name
        self.before_shape = None
        self.after_shape = None

    def fit(self, df: pd.DataFrame):
        self.before_shape = df.shape
        return self

    def transform(self, df: pd.DataFrame):
        df = df.rename({self.col: self.name}, axis=1)
        self.after_shape = df.shape
        return df


class CopyColumn(TransformerMixin):
    def __init__(self, from_col: str, to_col: str):
        self.from_col = from_col
        self.to_col = to_col
        self.before_shape = None
        self.after_shape = None

    def fit(self, df: pd.DataFrame, y: pd.DataFrame = None):
        self.before_shape = df.shape
        return self

    def transform(self, df: pd.DataFrame, y: pd.DataFrame = None):
        self.after_shape = df.shape
        df[self.to_col] = df[self.from_col]
        return df


class CopyLabels(TransformerMixin):
    def __init__(self):
        self.y_copy = None

    def fit(self, df: pd.DataFrame, y: pd.DataFrame):
        return self

    def transform(self, df: pd.DataFrame, y: pd.DataFrame):
        self.y_copy = y.copy()
        return df
