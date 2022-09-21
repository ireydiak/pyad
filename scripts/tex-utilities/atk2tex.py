import os
import re
import pandas as pd
import jinja2

from argparse import Namespace
from jsonargparse import ArgumentParser
from typing import List


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--fname", type=str, default="results.csv", help="name of the results file"
    )
    parser.add_argument(
        "--folders", help="path to result folders", type=List[str]
    )
    parser.add_argument(
        "--template", type=str, default="./examples/table_tmpl.tex", help="path to the latex/jinja template"
    )
    parser.add_argument(
        "--out", type=str, default="output/output.tex", help="name of the output file"
    )
    parser.add_argument(
        "--columns", type=List[str], help="dataframe columns to display", default=None
    )
    parser.add_argument(
        "--models", type=List[str], help="models to include (defaults to every model)", default=None
    )
    parser.add_argument(
        "--table_caption", type=str, default="table caption"
    )
    parser.add_argument(
        "--table_label", type=str, default="table label"
    )
    return parser.parse_args()


def create_template(
        path_to_template: str,
        datasets: List[str],
        headers: List[str],
        structure: dict,
        caption: str,
        label: str,
        column_positions: str
) -> str:
    latex_jinja_env = jinja2.Environment(
        block_start_string="\BLOCK{",
        block_end_string="}",
        variable_start_string="\VAR{",
        variable_end_string="}",
        comment_start_string="\#{",
        comment_end_string="}",
        line_statement_prefix="%%",
        line_comment_prefix="%#",
        trim_blocks=True,
        autoescape=False,
        loader=jinja2.FileSystemLoader(os.path.abspath("."))
    )
    template = latex_jinja_env.get_template(path_to_template)
    buf = template.render(
        datasets=datasets,
        headers=headers,
        structure=structure,
        caption=caption,
        label=label,
        column_positions=column_positions
    )
    return buf


def initialize_structure(attack_types: List[str], datasets: List[str]) -> dict:
    """

    Returns
    -------
    attack_type:
        dataset:
            [model_1, ..., model_n]
    """
    structure = {
        attack_type: {
            dataset: [] for dataset in datasets
        } for attack_type in attack_types
    }
    return structure


def csv2tex(
        fname: str,
        out_fname: str,
        path_to_tmpl: str,
        from_folders: List[str],
        cols: List[str],
        table_caption: str,
        table_label: str,
        models_to_include: List[str] = None
) -> None:
    # setup
    to_process, datasets = [], []
    headers = ["Traffic type", "Datasets"]
    datasets = [p.split(os.path.sep)[-1] for p in from_folders]
    structure, models = {}, set()
    # files to process
    for base_root in from_folders:
        assert os.path.exists(base_root), f"path {base_root} does not exist, aborting"
        datasets.append(base_root.split(os.path.sep)[-1])
        for model_name in os.listdir(base_root):
            if model_name in set(models_to_include):
                to_process.append(
                    os.path.join(base_root, model_name, fname)
                )
    # contains the data that will be displayed
    structure = initialize_structure(cols, datasets)
    # read csv files
    for path_to_file in to_process:
        # get name of model and dataset
        model_name = path_to_file.split(os.path.sep)[-2]
        dataset_name = path_to_file.split(os.path.sep)[-3]
        if os.path.exists(path_to_file):
            # load dataframe
            df = pd.read_csv(path_to_file)
            # read last line
            cols = sorted(list(set(df.columns) & set(cols)))
            data = df.loc[len(df) - 1, cols]
            models.add(model_name)
            # add average and standard deviation to structure
            for attack_name, score in data.items():
                avg = score.split("(")[0].strip()
                std = re.search(r'\((.*?)\)', score).group(1)
                structure[attack_name][dataset_name].append(
                    {"avg": avg, "std": std}
                )
        else:
            for k in structure.keys():
                structure[k][dataset_name].append({"avg": 0.0, "std": 0.0})
    headers.extend(
        sorted(list(models))
    )
    buf = create_template(
        path_to_tmpl, list(datasets), headers, structure,
        caption=table_caption,
        label=table_label,
        column_positions="l" * (len(headers) + 2)
    )
    with open(out_fname, "w") as f:
        f.write(buf)


def main():
    args = parse_args()
    csv2tex(
        fname=args.fname,
        out_fname=args.out,
        path_to_tmpl=args.template,
        from_folders=args.folders,
        cols=args.columns,
        table_caption=args.table_caption,
        table_label=args.table_label,
        models_to_include=args.models
    )


if __name__ == "__main__":
    main()
