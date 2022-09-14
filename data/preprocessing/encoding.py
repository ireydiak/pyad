import pandas as pd
from sklearn.base import TransformerMixin


class BinaryEncoder(TransformerMixin):
    def __init__(self, normal_label: str):
        self.normal_label = normal_label
        self.y = None

    def fit(self, df: pd.DataFrame, y: pd.DataFrame = None):
        return self

    def transform(self, df: pd.DataFrame):
        # Convert column to binary labels
        df[df == self.normal_label] = 0
        df[df != 0] = 1
        return df
