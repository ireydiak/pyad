from argparse import Namespace
from collections import defaultdict
from typing import List, Tuple
import os
import pandas as pd
import jsonargparse
import numpy as np

from scipy.io import loadmat
from sklearn.pipeline import Pipeline
from data.preprocessing import BinaryEncoder, CleanUnique, CleanNegative, CopyColumn, ReplaceNaN, CleanNaN


def parse_args() -> Namespace:
    parser = jsonargparse.ArgumentParser()

    # Paths
    parser.add_argument(
        "-d", "--path", type=str,
        help="Absolute path to original CSV file or path to root directory containing CSV files."
    )
    parser.add_argument(
        "-o", "--output_path", type=str,
        help="Path to the output directory. Folders will be added to this directory."
    )
    parser.add_argument(
        "--output_name", type=str,
        default="processed.csv",
        help="Output CSV file name."
    )

    parser.add_argument(
        "--nan_atol", type=float,
        default=0.05,
        help="Ratio of NaN values tolerated before dropping the columns"
    )
    parser.add_argument(
        "--negative_atol", type=float,
        default=0.05,
        help="Ratio of negative values tolerated before dropping the columns"
    )
    parser.add_argument(
        "--drop_cols", type=List[str],
        default=None,
        help="The name of the column(s) to be deleted."
    )

    return parser.parse_args()


class DataProcess:
    def __init__(self):
        pass

    def merge_step(
            self,
            input_path: str,
            output_path: str,
            drop_cols: List[str]
    ) -> Tuple[pd.DataFrame, dict]:
        df = pd.DataFrame()
        if os.path.isdir(input_path):
            for f in os.listdir(input_path):
                if f.endswith(".csv"):
                    path_to_csv = os.path.join(input_path, f)
                    print(f"processing {path_to_csv} ...")
                    chunk = pd.read_csv(
                        path_to_csv
                    )
                    chunk.columns = chunk.columns.str.strip()
                    chunk = chunk.drop(drop_cols, axis=1, errors="ignore")
                    df = pd.concat((df, chunk))
                else:
                    print(f"skipping file {f}")
            df.to_csv(
                os.path.join(output_path, "merged.csv"), index=False
            )
        else:
            df = pd.read_csv(input_path)
            df = df.drop(drop_cols, axis=1, errors="ignore")

        stats = {"in_features": df.shape[1], "n_instances": df.shape[0]}
        return df, stats

    def init_stats(self) -> dict:
        stats = defaultdict()
        stats["dropped_cols"] = ""
        stats["n_dropped_cols"] = 0
        stats["n_dropped_rows"] = 0

        return stats

    def clean_unique(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict, int, int]:
        stats = defaultdict()
        dropped_cols = 0
        unique_cols = df.columns[df.nunique() <= 1].tolist()
        stats["n_unique_cols"] = len(unique_cols)

        if unique_cols:
            print(f"Found {len(unique_cols)} column(s) with unique values: {unique_cols}")
            stats["unique_cols"] = ", ".join([str(col) for col in unique_cols])
            df = df.drop(unique_cols, axis=1)
            dropped_cols += len(unique_cols)
            unique_cols = df.columns[df.nunique() <= 1].tolist()

        assert len(unique_cols) == 0, f"There are still {len(unique_cols)} columns with unique values: {unique_cols}"

        return df, stats, 0, dropped_cols

    def clean_invalid(
            self,
            df: pd.DataFrame,
            label_col: str,
            normal_label: str,
            atol: float = 0.05
    ) -> Tuple[pd.DataFrame, dict, int, int]:
        # Verbose function
        stats = defaultdict()
        dropped_cols = []
        dropped_rows = 0

        # Replacing INF values with NaN
        df = df.replace([-np.inf, np.inf], np.nan)
        nan_cols = df.columns[df.isna().sum() > 0].tolist()
        stats["n_nan_cols"] = len(nan_cols)

        if nan_cols:
            stats["nan_cols"] = ", ".join([str(col) for col in nan_cols])
            print(f"Found {len(nan_cols)} column(s) with NaN values: {nan_cols}")

        # Drop column if the invalid ratio exceeds 5%
        for col in nan_cols:
            ratio = df.loc[:, col].isna().sum() / len(df)
            if ratio > atol:
                print(f"Dropping {col} with NaN ratio={ratio}>{atol}")
                df = df.drop(col, axis=1)
                dropped_cols.append(col)
            else:
                to_drop = df[df[col].isna()]
                assert to_drop[label_col != normal_label].sum() == 0, "found NaNs within abnormal data, aborting"
                print(f"Dropping {len(to_drop)} rows")
                dropped_rows += len(to_drop)
                df = df.drop(to_drop.index, axis=0)
        remaining_nans = df.isna().sum().sum()
        assert remaining_nans == 0, f"There are still {remaining_nans} NaN values"

        if dropped_cols:
            print(f"Dropped columns with NaN-ratio >{atol}%: {dropped_cols}")

        return df, stats, dropped_rows, len(dropped_cols)

    def clean_negative(
            self,
            df: pd.DataFrame,
            label_col: str,
            normal_label: str,
            no_negative: List[str] = None,
            atol: float = 0.05
    ) -> Tuple[pd.DataFrame, dict, int, int]:
        negative_columns = no_negative or df.columns[
            df.loc[:, :].lt(0).sum() > 0
            ]
        dropped_cols = []
        dropped_rows = 0
        stats = defaultdict()

        for col in negative_columns:
            ratio = df.loc[:, col].lt(0).sum() / len(df)
            if ratio > atol:
                df = df.drop(col, axis=1)
                dropped_cols.append(col)
            else:
                to_drop = df[~df[col].lt(0)]
                dropped_rows += len(to_drop)
                assert to_drop[label_col != normal_label].sum() == 0, "found NaNs within abnormal data, aborting"
                df = df.drop(to_drop.index, axis=0)
                df.drop(to_drop, axis=0, inplace=True)

        df = df.drop(dropped_cols, axis=1)
        stats["negative_columns"] = len(dropped_cols)

        return df, stats, dropped_rows, len(dropped_cols)

    def update_stats(
            self,
            stats: dict,
            step_stats: dict,
            dropped_rows: int,
            dropped_cols: int
    ) -> dict:
        stats = dict(**stats, **step_stats)
        stats["n_dropped_cols"] += dropped_cols
        stats["n_dropped_rows"] += dropped_rows
        return stats

    def save_stats(self, path: str, *stats: dict) -> None:
        vals = {k: v for d in stats for k, v in d.items()}
        with open(path, 'w') as f:
            f.write(','.join(vals.keys()) + '\n')
            f.write(','.join([str(val) for val in vals.values()]))

    def process(
            self,
            input_path: str,
            label_col: str,
            normal_label: str,
            nan_atol: float,
            negative_atol: float,
            no_negative: List[str],
            output_path: str,
            output_name: str,
            drop_cols: List[str] = None
    ) -> None:
        stats = self.init_stats()

        # Load/merge files
        df, step_stats = self.merge_step(input_path, output_path, drop_cols)
        stats = self.update_stats(stats, step_stats, 0, 0)

        # Manage columns with unique values
        df, uniq_stats, dropped_rows, dropped_cols = self.clean_unique(df)
        stats = self.update_stats(stats, uniq_stats, dropped_rows, dropped_cols)

        # Manage columns with invalid values
        df, uniq_stats, dropped_rows, dropped_cols = self.clean_invalid(
            df,
            label_col,
            normal_label,
            atol=nan_atol
        )
        stats = self.update_stats(stats, uniq_stats, dropped_rows, dropped_cols)

        # Manage columns with negative values
        df, negative_stats, dropped_rows, dropped_cols = self.clean_negative(
            df,
            label_col,
            normal_label,
            no_negative,
            atol=negative_atol
        )
        stats = self.update_stats(stats, negative_stats, dropped_rows, dropped_cols)

        # Copy labels and binarize labels
        # Keep the full-labels aside before "binarizing" them
        df["Category"] = df[label_col]
        df.rename({label_col: "Label"}, axis=1, inplace=True)
        # Convert labels to binary labels
        df.loc[df["Label"].isin(normal_label), "Label"] = 0
        df.loc[df["Label"] != 0, "Label"] = 1

        # Save DataFrame and statistics
        df.to_csv(
            os.path.join(output_path, f"{output_name}.csv")
        )
        self.save_stats(
            os.path.join(output_path, f"{output_name}_info.csv"),
            stats
        )


# def main() -> None:
#     args = parse_args()
#     processor = DataProcess()
#     processor.process(
#         input_path=args.input_path,
#         label_col=args.label_col,
#         normal_label=args.normal_label,
#         nan_atol=args.nan_atol,
#         negative_atol=args.negative_atol,
#         no_negative=args.no_negative,
#         output_path=args.output_path,
#         output_name=args.output_name,
#         drop_cols=args.drop_cols
#     )

class IDSPipeline:
    def __init__(self, path: str, output_path: str, output_name: str, drop_cols: List[int] = None):
        self.path = path
        self.drop_cols = drop_cols or []
        self.output_path = output_path
        self.output_name = output_name

    def uniformize_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        # Group DoS attacks
        mask = df["Label"].str.startswith("DoS")
        df.loc[mask, "Label"] = "DoS"
        # Group Web attacks
        mask = df["Label"].str.startswith("Web Attack")
        df.loc[mask, "Label"] = "Web Attack"
        # Rename attacks to match the labels of IDS2018
        # Rename BENIGN to Benign
        mask = df["Label"].str.match("BENIGN")
        df.loc[mask, "Label"] = "Benign"
        # Rename FTP-Patator to FTP-BruteForce
        mask = df["Label"].str.match("FTP-Patator")
        df.loc[mask, "Label"] = "FTP-BruteForce"
        # Rename SSH-Patator to SSH-Bruteforce
        mask = df["Label"].str.match("SSH-Patator")
        df.loc[mask, "Label"] = "SSH-Bruteforce"
        return df

    def load_data(self):
        df = pd.DataFrame()
        if os.path.isdir(self.path):
            for f in os.listdir(self.path):
                if f.endswith(".csv"):
                    path_to_csv = os.path.join(self.path, f)
                    print(f"processing {path_to_csv} ...")
                    chunk = pd.read_csv(path_to_csv)
                    chunk.columns = chunk.columns.str.strip()
                    chunk = chunk.drop(self.drop_cols, axis=1, errors="ignore")
                    df = pd.concat((df, chunk))
                else:
                    print(f"skipping file {f}")
            df.to_csv(
                os.path.join(self.output_path, "merged.csv"), index=False
            )
        else:
            df = pd.read_csv(self.path)
            df = df.drop(self.drop_cols, axis=1, errors="ignore")
        df = self.uniformize_labels(df)
        return df

    def process(self):
        df = self.load_data()
        pipeline = Pipeline(
            steps=[
                ("Clean Unique", CleanUnique()),
                ("Simple NaN Imputer", ReplaceNaN(missing_values=np.nan, fill_value=0)),
                ("Clean Negative", CleanNegative(atol=0.01, label_col="Label", normal_label="Benign")),
                ("Copy Column", CopyColumn(from_col="Label", to_col="Category")),
                ("Binary Encoding", BinaryEncoder(col="Label", normal_label="Benign"))
            ],
        )
        df = pipeline.fit_transform(df)
        df.to_csv(
            os.path.join(self.output_path, self.output_name)
        )


class MATPipeline:
    def __init__(self, path: str, output_path: str, output_name: str, drop_cols: List[int] = None):
        self.path = path
        self.drop_cols = drop_cols or []
        self.output_path = output_path
        self.output_name = output_name

    def load_data(self):
        mat = loadmat(self.path)
        X = mat['X']  # variable in mat file
        y = mat['y']
        # now make a data frame, setting the time stamps as the index
        df = pd.DataFrame(
            np.concatenate((X, y), axis=1),
            columns=list(np.arange(1, X.shape[1] + 1)) + ["Label"]
        )
        return df

    def process(self):
        df = self.load_data()
        pipeline = Pipeline(
            steps=[
                ("Clean Unique", CleanUnique()),
                ("Clean NaN", CleanNaN(atol=0.01, label_col="Label", normal_label=0)),
                ("Clean Negative", CleanNegative(atol=0.01, label_col="Label", normal_label=0)),
            ],
        )
        df = pipeline.fit_transform(df)
        df.to_csv(
            os.path.join(self.output_path, self.output_name)
        )


def main():
    args = parse_args()
    pipeline = IDSPipeline(
        path=args.path,
        output_path=args.output_path,
        output_name=args.output_name,
        drop_cols=args.drop_cols
    )
    pipeline.process()


if __name__ == "__main__":
    main()
