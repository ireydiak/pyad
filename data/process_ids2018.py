from argparse import Namespace
from typing import Tuple
import warnings
import os

import jsonargparse
import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings('ignore')

NORMAL_LABEL = 0
NORMAL_CAT = "Benign"
ANORMAL_LABEL = 1


def parse_args() -> Namespace:
    parser = jsonargparse.ArgumentParser()
    parser.add_argument(
        '-d', '--path', type=str,
        default='',
        help='Path to original CSV file or path to root directory containing CSV files'
    )
    parser.add_argument(
        '-o', '--export_path', type=str,
        help='Path to the output directory. Folders will be added to this directory.'
    )
    args = parser.parse_args()
    return args


def merge_step(path_to_files: str) -> Tuple[pd.DataFrame, dict]:
    chunks, chunk = [], None
    stats = defaultdict()
    df = pd.DataFrame()

    if os.path.isdir(path_to_files):
        for f in os.listdir(path_to_files):
            chunk = pd.read_csv(os.path.join(path_to_files, f))
            chunk.columns = chunk.columns.str.strip()
            df = pd.concat((df, chunk))
            print(f)
    else:
        df = pd.read_csv(path_to_files)
    stats["dropped_cols"] = ""
    stats["n_dropped_cols"] = 0
    stats["n_dropped_rows"] = 0
    stats["n_instances"] = len(df)
    stats["n_features"] = df.shape[1] - 1
    stats["anomaly_ratio"] = "{:2.4f}".format(
        (df["Label"] != NORMAL_CAT).sum() / len(df)
    )

    return df, stats


def uniformize_labels(df: pd.DataFrame) -> pd.DataFrame:
    # Group DoS attacks
    mask = df["Label"].str.startswith("DoS")
    df.loc[mask, "Label"] = "DoS"
    # Group DDoS attacks
    mask = df["Label"].str.startswith("DDoS")
    df.loc[mask, "Label"] = "DDoS"
    mask = df["Label"].str.startswith("DDOS")
    df.loc[mask, "Label"] = "DDoS"
    # Group Web attacks
    mask = df["Label"].str.startswith("Brute Force")
    df.loc[mask, "Label"] = "Web Attack"
    mask = df["Label"].str.startswith("SQL")
    df.loc[mask, "Label"] = "Web Attack"
    # Rename Infilteration to Infiltration
    mask = df["Label"].str.match("Infilteration")
    df.loc[mask, "Label"] = "Infiltration"

    return df


def clean_uniq(df: pd.DataFrame, stats: dict) -> Tuple[pd.DataFrame, dict]:
    # unique values
    uniq_cols = df.columns[df.nunique() <= 1].tolist()
    stats["n_unique_cols"] = len(uniq_cols)
    if uniq_cols:
        print("Found {} columns with unique values: {}".format(len(uniq_cols), uniq_cols))
        stats["unique_cols"] = ", ".join([str(col) for col in uniq_cols])
        df.drop(uniq_cols, axis=1, inplace=True)
        stats["n_dropped_cols"] += len(uniq_cols)
        uniq_cols = df.columns[df.nunique() <= 1].tolist()
    assert len(uniq_cols) == 0, "Found {} columns with unique values: {}".format(len(uniq_cols), uniq_cols)
    print("Columns are valid with more than one distinct value")
    return df, stats


def clean_invalid(df: pd.DataFrame, stats: dict) -> Tuple[pd.DataFrame, dict]:
    # nan values
    # Replacing INF values with NaN
    df = df.replace([-np.inf, np.inf], np.nan)
    nan_cols = df.columns[df.isna().sum() > 0].tolist()
    stats["n_nan_cols"] = len(nan_cols)
    if nan_cols:
        stats["nan_cols"] = ", ".join([str(col) for col in nan_cols])
    print("Found NaN columns: {}".format(nan_cols))

    # replace invalid values
    n_nan_values = df[nan_cols].isna().sum().sum()
    ratio = (df[nan_cols].isna().sum()[0] / len(df)) * 100
    print(df[nan_cols].isna().sum() / len(df))
    df = df.fillna(0)
    stats["replaced_nan_values"] = "{} instances or {:2.4f}%".format(n_nan_values, ratio)
    print("Replaced {:2.4f}% of original data".format(ratio))
    remaining_nans = df.isna().sum().sum()
    assert remaining_nans == 0, "There are still {} NaN values".format(remaining_nans)
    remaining_inf = df.isinf().sum().sum()
    assert remaining_inf == 0, "There are still {} INF values".format(remaining_inf)
    return df, stats


def clean_negative(df: pd.DataFrame, stats: dict) -> Tuple[pd.DataFrame, dict]:
    n_anom_before = (df["Label"] != NORMAL_CAT).sum()
    # select numerical columns
    num_cols = df.select_dtypes(exclude="object").columns
    # create mask for negative values on numerical columns
    mask = (df[num_cols] < 0).sum() > 0
    # select the numerical columns with negative values
    neg_cols = df[num_cols].columns[mask]
    stats["n_negative_cols"] = len(neg_cols)
    stats["negative_cols"] = ", ".join(neg_cols)
    print("Found {} columns with negative values: {}".format(len(neg_cols), neg_cols))
    # remove columns with negative values and associated with attacks
    neg_cols_when_anomalies = df[num_cols].columns[
        (df[num_cols][((df[num_cols]).any(1)) & (df["Label"] != NORMAL_CAT)] < 0).sum() > 0
    ]
    to_drop = list(neg_cols_when_anomalies)
    stats["n_dropped_cols"] += len(neg_cols_when_anomalies)
    stats["dropped_cols"] = stats["dropped_cols"] + ", ".join(to_drop)
    df = df.drop(to_drop, axis=1)
    print("Dropped {} columns {} (negative values with anomalies)".format(len(to_drop), to_drop))
    # remove remaining negative rows exclusively associated with `Benign` traffic
    df = df.reset_index()
    num_cols = df.select_dtypes(include=np.number).columns
    idx_to_drop = df[(df[num_cols] < 0).any(1)].index
    # weird hack to go around an annoying index behavior from pandas
    # selecting the index from the subset `num_cols` includes anomalies on the complete dataframe
    # hence, to avoid deleting attacks, we compute the intersection between normal data and remaining negative values
    idx_to_drop = list(set(df[(df.Label == NORMAL_CAT)].index) & set(idx_to_drop))
    n_dropped = len(idx_to_drop)
    stats["n_dropped_rows"] += n_dropped
    df = df.drop(idx_to_drop, axis=0)
    print("Dropped {} rows (negative values for normal data)".format(n_dropped))
    assert (df[num_cols] < 0).any(1).sum() == 0, "There are still negative values"
    print("There are no more negative values")
    n_anom_after = (df["Label"] != NORMAL_CAT).sum()
    assert n_anom_before == n_anom_after, "dropped {} anomalies, aborting".format(n_anom_before - n_anom_after)
    return df, stats


def clean_step(df: pd.DataFrame, stats: dict):
    # 1- uniformize anomaly labels
    df = uniformize_labels(df)
    # 2- drop categorical attributes and invalid rows
    to_drop = ["Dst IP", "Flow ID", "Src IP", "Src Port", "Flow Duration", "Protocol", "Timestamp", "Dst Port"]
    df = df.drop(to_drop, axis=1, errors="ignore")
    #    remove 59 rows that are duplicates of the headers
    df = df.drop(df[df.Label == "Label"].index, axis=0)
    # 3- remove columns with unique values
    df, stats = clean_uniq(df, stats)
    # 4- manage NaN/INF and other invalid values
    df, stats = clean_invalid(df, stats)
    # Cast types
    types = {col_name: np.float32 if col_name != "Label" else "object" for col_name in df.columns}
    df = df.astype(types)
    # 5- negative values
    df, stats = clean_negative(df, stats)

    # Keep the full-labels aside before "binarizing" them
    df["Category"] = df["Label"]
    # Convert labels to binary labels
    df.loc[df["Label"] == NORMAL_CAT, "Label"] = 0
    df.loc[df["Label"] != 0, "Label"] = 1

    return df, stats


def process(path: str, export_path: str):
    # 1- Merge the different CSV files into one dataframe
    df, stats = merge_step(path)

    # 2- Clean the data (remove invalid rows and columns, etc.)
    df, stats = clean_step(df, stats)

    # Save info about cleaning step
    save_stats(
        os.path.join(export_path, "ids2018_info.csv"), stats
    )
    # Save final dataframe
    df.to_csv(
        os.path.join(export_path, "ids2018.csv")
    )


def save_stats(path: str, *stats: dict) -> None:
    vals = {k: v for d in stats for k, v in d.items()}
    with open(path, 'w') as f:
        f.write(','.join(vals.keys()) + '\n')
        f.write(','.join([str(val) for val in vals.values()]))


def main():
    args = parse_args()
    process(
        path=args.path,
        export_path=args.export_path
    )


if __name__ == '__main__':
    main()
