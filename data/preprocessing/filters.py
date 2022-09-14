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

    def fit(self, df: pd.DataFrame, y: pd.DataFrame):
        self.before_shape = df.shape
        unique_cols = df.columns[df.nunique() <= 1].tolist()

        if unique_cols:
            print(f"Found {len(unique_cols)} column(s) with unique values: {unique_cols}")
            self.dropped_columns = unique_cols

        return self

    def transform(self, df: pd.DataFrame):
        df = df.drop(self.dropped_columns, axis=1)
        self.after_shape = df.shape
        unique_cols = df.columns[df.nunique() <= 1].tolist()
        assert len(unique_cols) == 0, f"There are still {len(unique_cols)} columns with unique values: {unique_cols}"
        return df


class CleanNegative(TransformerMixin):
    def __init__(self, atol: float, normal_label, selected_columns: List[str] = None):
        self.selected_columns = selected_columns
        self.atol = atol
        self.normal_label = normal_label
        self.before_shape = None
        self.after_shape = None
        self.dropped_columns = []
        self.dropped_rows = []
        self.summary = {
            "affected_columns": [],
            "dropped_columns": [],
            "n_dropped_rows": 0,
            "n_dropped_columns": 0
        }

    def fit(self, df: pd.DataFrame, y: pd.DataFrame):
        self.before_shape = df.shape
        self.selected_columns = self.selected_columns or df.columns
        dropped_cols, dropped_rows = [], set()

        neg_cols = df.columns[
            df.loc[:, self.selected_columns].lt(0).sum() > 0
        ].tolist()
        if neg_cols:
            print(f"Found {len(neg_cols)} column(s) with negative values: {neg_cols}")

        self.summary["affected_columns"] = neg_cols

        # Drop column if the invalid ratio exceeds `atol`
        for col in neg_cols:
            ratio = df.loc[:, col].lt(0).sum() / len(df)
            rows_to_drop = df[df[col].lt(0)]
            n_anomalies = len(
                set(y[y != self.normal_label].index) & set(rows_to_drop.index)
            )
            self.summary[col] = dict(
                ratio=float(ratio),
                n_anomalies=n_anomalies,
                n_affected_rows=len(rows_to_drop),
                n_dropped_rows=0,
                dropped=False
            )
            if ratio > self.atol or n_anomalies > 0:
                print(f"Dropped {col} with negative ratio={ratio}>{self.atol} and {n_anomalies} anomalies")
                dropped_cols.append(col)
            else:
                dropped_rows = dropped_rows | set(rows_to_drop.index.tolist())

        # journaling
        self.dropped_columns = dropped_cols
        self.dropped_rows = list(dropped_rows)
        self.summary["n_dropped_columns"] = len(dropped_cols)
        self.summary["n_dropped_rows"] = len(dropped_rows)

        # update selected columns attribute
        self.selected_columns = list(
            set(self.selected_columns) - set(self.dropped_columns)
        )
        return self

    def transform(self, df: pd.DataFrame):
        # apply changes
        df = df.drop(self.dropped_columns, axis=1)
        df = df.drop(self.dropped_rows, axis=0)
        # double-check there are no more negative columns
        remaining_negatives = df.loc[:, self.selected_columns].lt(0).sum().sum()
        assert remaining_negatives == 0, f"There are still {remaining_negatives} negative values"
        # journaling
        self.after_shape = df.shape
        return df


class CleanNaN(TransformerMixin):
    def __init__(self, atol: float, normal_label):
        self.atol = atol
        self.normal_label = normal_label
        self.before_shape = None
        self.after_shape = None
        self.dropped_columns = []
        self.dropped_rows = []
        self.affected_columns = []
        self.summary = {
            "affected_columns": [],
            "dropped_columns": []
        }

    def fit(self, df: pd.DataFrame, y: pd.DataFrame):
        # setup
        self.before_shape = df.shape
        dropped_cols, dropped_rows = [], set()
        # get columns with NaN values
        nan_cols = df.columns[df.isna().sum() > 0].tolist()
        self.affected_columns = nan_cols

        # Drop column if the invalid ratio exceeds `atol`
        for col in nan_cols:
            ratio = df.loc[:, col].isna().sum() / len(df)
            rows_to_drop = df[df[col].isna()]
            n_anomalies = (y.iloc[df.index] != self.normal_label).sum()
            self.summary[col] = dict(
                ratio=float(ratio),
                n_anomalies=n_anomalies,
                n_affected_rows=len(rows_to_drop),
                dropped=False
            )
            if ratio > self.atol or n_anomalies > 0:
                print(f"Dropped {col} with nan ratio={ratio}>{self.atol} and {n_anomalies} anomalies")
                dropped_cols.append(col)
                self.summary[col]["dropped"] = True
            else:
                assert n_anomalies == 0, f"operation would remove {n_anomalies} anomalies, aborting"
                print(f"Dropped {len(rows_to_drop)} rows")
                dropped_rows = dropped_rows | set(rows_to_drop.index.tolist())

        self.dropped_columns = dropped_cols
        self.dropped_rows = list(dropped_rows)
        self.summary["n_dropped_columns"] = len(dropped_cols)
        self.summary["n_dropped_rows"] = len(dropped_rows)

        return self

    def transform(self, df: pd.DataFrame):
        # apply changes
        df = df.drop(self.dropped_columns, axis=1)
        df = df.drop(self.dropped_rows, axis=0)
        # double-check there are no more NaNs
        remaining_nans = df.isna().sum().sum()
        assert remaining_nans == 0, f"There are still {remaining_nans} NaN values"
        # journaling
        self.after_shape = df.shape
        return df


class ReplaceNaN(TransformerMixin):
    def __init__(self, missing_values=np.nan, fill_value=0):
        self.missing_values = missing_values
        self.fill_value = fill_value
        self.before_shape = None
        self.after_shape = None
        self.dropped_columns = []
        self.dropped_rows = []
        self.ratio_df = None
        self.summary = {
            "affected_columns": [],
            "n_affected_rows": 0,
            "dropped_columns": []
        }

    def fit(self, df: pd.DataFrame, y: pd.DataFrame):
        self.before_shape = df.shape
        # Replacing INF values with NaN
        df = df.replace([-np.inf, np.inf], np.nan)
        nan_cols = df.columns[df.isna().sum() > 0].tolist()
        self.summary["affected_columns"] = nan_cols
        print(f"Found {len(nan_cols)} column(s) with NaN values: {nan_cols}")
        # keep track of changes
        for col in nan_cols:
            n_nan = df.loc[:, col].isna().sum()
            ratio = (n_nan / len(df)) * 100
            self.summary[col] = dict(
                n_affected_rows=int(n_nan),
                ratio=float(ratio)
            )
        # journaling
        self.summary["n_affected_rows"] = int(df.isna().sum().sum())
        return self

    def transform(self, df: pd.DataFrame):
        # apply changes
        df = df.fillna(self.fill_value)
        # double-check there are no more NaNs
        remaining_nans = df.isna().sum().sum()
        assert remaining_nans == 0, f"There are still {remaining_nans} NaN values"
        # journaling
        self.after_shape = df.shape
        return df
