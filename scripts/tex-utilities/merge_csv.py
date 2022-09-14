from argparse import Namespace
from typing import List

import os
import pandas as pd
import jsonargparse


def parse_args() -> Namespace:
    parser = jsonargparse.ArgumentParser()
    parser.add_argument(
        "--directory", type=str, help="base directory where csv files reside"
    )
    parser.add_argument(
        "--filename", type=str, help="csv filename, must be the same across directories"
    )
    parser.add_argument(
        "--columns", type=List[str], help="csv columns to keep", default=None
    )
    parser.add_argument(
        "--out", type=str, help="name of the output file", default=None
    )
    return parser.parse_args()


def merge_csv(
        directory: str,
        filename: str,
        columns: List[str] = None
) -> pd.DataFrame:
    paths = [
        os.path.join(directory, dr, filename) for dr in os.listdir(directory) if os.path.isdir(os.path.join(directory, dr))
    ]
    data = []
    indexes = []
    for p in paths:
        df = pd.read_csv(p)
        model_name = p.split(os.path.sep)[-2]
        indexes.append(model_name)
        if columns is not None:
            row = df.loc[len(df) - 1, columns].to_list()
        else:
            row = df.loc[len(df) - 1].to_list()
        data.append(row)
    df = pd.DataFrame(data=data, index=indexes, columns=columns)
    return df


def main() -> None:
    args = parse_args()
    df = merge_csv(
        directory=args.directory,
        filename=args.filename,
        columns=args.columns,
    )
    fname = args.out or "merge.csv"
    df.to_csv(fname)


if __name__ == "__main__":
    main()
