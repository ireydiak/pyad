from argparse import Namespace
from typing import List
import jsonargparse
from data.preprocessing.pipeline import IDSPipeline, MATPipeline


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
        "--output_fname", type=str,
        default="processed.csv",
        help="Output CSV file name."
    )

    parser.add_argument(
        "--nan_atol", type=float,
        default=0.01,
        help="Ratio of NaN values tolerated before dropping the columns"
    )
    parser.add_argument(
        "--negative_atol", type=float,
        default=0.01,
        help="Ratio of negative values tolerated before dropping the columns"
    )
    parser.add_argument(
        "--drop_cols", type=List[str],
        default=None,
        help="The name of the column(s) to be deleted."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = IDSPipeline(
        path=args.path,
        output_path=args.output_path,
        output_name=args.output_fname,
        drop_cols=args.drop_cols
    )
    pipeline.process()


if __name__ == "__main__":
    main()
