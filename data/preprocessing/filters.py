import pandas as pd
import numpy as np

from typing import List
from sklearn.base import TransformerMixin


class CleanUnique(TransformerMixin):
    def __init__(self):
        self.before_shape = None
        self.after_shape = None
        self.dropped_columns = []
        self.dropped_rows = []

    def fit(self, df: pd.DataFrame, y: pd.DataFrame = None):
        self.before_shape = df.shape
        unique_cols = df.columns[df.nunique() <= 1].tolist()

        if unique_cols:
            print(f"Found {len(unique_cols)} column(s) with unique values: {unique_cols}")
            df = df.drop(unique_cols, axis=1)
            self.dropped_columns = unique_cols
            unique_cols = df.columns[df.nunique() <= 1].tolist()

        assert len(unique_cols) == 0, f"There are still {len(unique_cols)} columns with unique values: {unique_cols}"

        return self

    def transform(self, df: pd.DataFrame, y: pd.DataFrame = None):
        df = df.drop(self.dropped_columns, axis=1)
        self.after_shape = df.shape
        return df


class CleanNegative(TransformerMixin):
    def __init__(self, atol: float, label_col: str, normal_label, selected_columns: List[str] = None):
        self.selected_columns = selected_columns
        self.atol = atol
        self.label_col = label_col
        self.normal_label = normal_label
        self.before_shape = None
        self.after_shape = None
        self.dropped_columns = []
        self.dropped_rows = []

    def fit(self, df: pd.DataFrame, y: pd.DataFrame = None):
        self.before_shape = df.shape
        self.selected_columns = self.selected_columns or df.columns
        self.selected_columns = list(
            set(self.selected_columns) - {self.label_col}
        )
        return self

    def transform(self, df: pd.DataFrame, y: pd.DataFrame = None):
        dropped_cols, dropped_rows = [], []

        neg_cols = df.columns[
            df.loc[:, self.selected_columns].lt(0).sum() > 0
        ].tolist()

        if neg_cols:
            print(f"Found {len(neg_cols)} column(s) with negative values: {neg_cols}")

        # Drop column if the invalid ratio exceeds `atol`
        for col in neg_cols:
            ratio = df.loc[:, col].lt(0).sum() / len(df)
            if ratio > self.atol:
                df = df.drop(col, axis=1)
                print(f"Dropped {col} with negative ratio={ratio}>{self.atol}")
                dropped_cols.append(col)
            else:
                to_drop = df[df[col].lt(0)]
                err_msg = "found negative values within abnormal data, aborting"
                assert to_drop[self.label_col != self.normal_label].sum() == 0, err_msg
                df = df.drop(to_drop.index, axis=0)
                print(f"Dropped {len(to_drop)} rows")
                dropped_rows.append(to_drop.index)

        remaining_negatives = df.loc[:, self.selected_columns].lt(0).sum().sum()
        assert remaining_negatives == 0, f"There are still {remaining_negatives} negative values"

        if len(dropped_cols) > 0:
            print(f"Dropped columns with negative-ratio >{self.atol}%: {dropped_cols}")

        self.after_shape = df.shape
        self.dropped_columns = dropped_cols
        self.dropped_rows = dropped_rows
        return df


class CleanNaN(TransformerMixin):
    def __init__(self, atol: float, label_col: str, normal_label):
        self.atol = atol
        self.label_col = label_col
        self.normal_label = normal_label
        self.before_shape = None
        self.after_shape = None
        self.dropped_columns = []
        self.dropped_rows = []
        self.affected_columns = []

    def fit(self, df: pd.DataFrame, y: pd.DataFrame = None):
        self.before_shape = df.shape
        return self

    def transform(self, df: pd.DataFrame, y: pd.DataFrame = None):
        dropped_cols, dropped_rows = [], []

        nan_cols = df.columns[df.isna().sum() > 0].tolist()
        self.affected_columns = nan_cols

        if nan_cols:
            print(f"Found {len(nan_cols)} column(s) with nan values: {nan_cols}")

        # Drop column if the invalid ratio exceeds `atol`
        for col in nan_cols:
            ratio = df.loc[:, col].isna().sum() / len(df)
            if ratio > self.atol:
                df = df.drop(col, axis=1)
                print(f"Dropped {col} with nan ratio={ratio}>{self.atol}")
                dropped_cols.append(col)
            else:
                to_drop = df[df[col].isna()]
                assert to_drop[
                           self.label_col != self.normal_label
                           ].sum() == 0, "found negative values within abnormal data, aborting"
                df = df.drop(to_drop.index, axis=0)
                print(f"Dropped {len(to_drop)} rows")
                dropped_rows.append(to_drop.index)

        remaining_nans = df.isna().sum().sum()
        assert remaining_nans == 0, f"There are still {remaining_nans} NaN values"

        if len(dropped_cols) > 0:
            print(f"Dropped columns with negative-ratio >{self.atol}%: {dropped_cols}")

        self.after_shape = df.shape
        self.dropped_columns = dropped_cols
        self.dropped_rows = dropped_rows
        return df


class ReplaceNaN(TransformerMixin):
    def __init__(self, missing_values=np.nan, fill_value=0):
        self.missing_values = missing_values
        self.fill_value = fill_value
        self.before_shape = None
        self.after_shape = None
        self.dropped_columns = []
        self.dropped_rows = []
        self.affected_columns = []
        self.ratio_df = None

    def fit(self, df: pd.DataFrame, y: pd.DataFrame = None):
        self.before_shape = df.shape
        return self

    def transform(self, df: pd.DataFrame, y: pd.DataFrame = None):
        # Replacing INF values with NaN
        df = df.replace([-np.inf, np.inf], np.nan)
        nan_cols = df.columns[df.isna().sum() > 0].tolist()
        self.affected_columns = nan_cols

        if nan_cols:
            print(f"Found {len(nan_cols)} column(s) with NaN values: {nan_cols}")
            self.ratio_df = pd.DataFrame(pd.concat((
                df[nan_cols].isna().sum(),
                (df[nan_cols].isna().sum() / len(df)) * 100
            ), axis=1))

        df = df.fillna(self.fill_value)

        remaining_nans = df.isna().sum().sum()
        assert remaining_nans == 0, f"There are still {remaining_nans} NaN values"

        self.after_shape = df.shape
        return df
