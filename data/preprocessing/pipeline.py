from typing import List, Tuple, Optional
import os
import pandas as pd
import numpy as np
import yaml

from scipy.io import loadmat
from sklearn.pipeline import Pipeline
from data.preprocessing import BinaryEncoder, CleanUnique, CleanNegative, CopyLabels, ReplaceNaN, CleanNaN


class BasePipeline:

    def __init__(self, path: str, output_path: str, output_name: str, drop_cols: List[int] = None):
        self.path = path
        self.drop_cols = drop_cols or []
        self.output_path = output_path
        self.output_name = output_name if output_name.endswith(".npz") else output_name + ".npz"

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass

    def setup_pipeline(self) -> Pipeline:
        pass

    def preprocess_labels(self, y: pd.DataFrame, dropped_indexes: List[int]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        y = y.drop(dropped_indexes, axis=0)
        return y.astype(np.int8).to_numpy(), None

    def process(self):
        summary = {}
        # Load data and labels
        df, y = self.load_data()
        summary["initial_n_instances"] = df.shape[0]
        summary["initial_in_features"] = df.shape[1]
        # Apply preprocessing on data
        data_pipeline = self.setup_pipeline()
        df = data_pipeline.fit_transform(df, y)
        # Fetch dropped indexes and steps summaries for the report
        dropped_indexes = set()
        for step_name, step in data_pipeline.steps:
            if hasattr(step, "summary"):
                summary[step_name] = step.summary
            if hasattr(step, "dropped_rows"):
                dropped_indexes = dropped_indexes | set(step.dropped_rows)
        summary["final_n_instances"] = df.shape[0]
        summary["final_in_features"] = df.shape[1]
        # Save changes summary in file
        info_fname = os.path.join(self.output_path, self.output_name.split(".")[0] + "_info.yaml")
        with open(info_fname, "w") as f:
            f.write(
                yaml.dump(summary, default_flow_style=False)
            )
        y, labels = self.preprocess_labels(y, dropped_indexes)

        # Save data
        np.savez(
            os.path.join(self.output_path, self.output_name),
            X=df.to_numpy(),
            y=y,
            labels=labels if labels is not None else []
        )
        # df.to_csv(
        #     os.path.join(self.output_path, self.output_name)
        # )


class IDSPipeline(BasePipeline):

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

    def setup_pipeline(self) -> Pipeline:
        data_pipeline = Pipeline(
            steps=[
                ("Clean Unique", CleanUnique()),
                ("NaN Imputer", ReplaceNaN(missing_values=np.nan, fill_value=0)),
                ("Negative Imputer", CleanNegative(atol=0.01, normal_label="Benign")),
                ("Copy Labels", CopyLabels(to_col="Category")),
            ]
        )
        return data_pipeline

    def preprocess_labels(self, y: pd.DataFrame, dropped_indexes: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        # Remove dropped indexes
        y = y.drop(list(dropped_indexes), axis=0)
        # Copy labels
        labels = y.copy()
        # Encode binary labels
        enc = BinaryEncoder(normal_label="Benign")
        y = enc.fit_transform(y)
        return y.astype(np.int8).to_numpy(), labels.to_numpy()

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
        return df.drop("Label", axis=1), df.loc[:, "Label"]


class MATPipeline(BasePipeline):

    def load_data(self):
        mat = loadmat(self.path)
        X = mat['X']  # variable in mat file
        y = mat['y']
        # now make a data frame, setting the time stamps as the index
        return pd.DataFrame(X), pd.DataFrame(y)
    
    def setup_pipeline(self) -> Pipeline:
        pipeline = Pipeline(
            steps=[
                ("Clean Unique", CleanUnique()),
                ("Clean NaN", CleanNaN(atol=0.01, normal_label=0)),
                ("Clean Negative", CleanNegative(atol=0.01, normal_label=0)),
            ],
        )
        return pipeline
