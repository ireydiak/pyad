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
        "--results_fname", type=str, default="results.csv", help="name of the results file"
    )
    parser.add_argument(
        "--from_folders", help="path to result folders", type=List[str]
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
        "--table_caption", type=str, default="table caption"
    )
    parser.add_argument(
        "--table_label", type=str, default="table label"
    )
    return parser.parse_args()


def create_template(
        path_to_template: str,
        datasets: List[str],
        metrics: List[str],
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
        metrics=metrics,
        structure=structure,
        caption=caption,
        label=label,
        column_positions=column_positions
    )
    return buf


def csv2tex(
        fname: str,
        out_fname: str,
        path_to_tmpl: str,
        from_folders: List[str],
        cols: List[str],
        table_caption: str,
        table_label: str
) -> None:
    to_process = []
    for base_root in from_folders:
        to_process.extend(list(
            map(lambda folder: os.path.join(base_root, folder, fname), os.listdir(base_root))
        ))
    metrics, structure = set(), {}
    datasets = set([ptf.split(os.path.sep)[-3] for ptf in to_process])
    structure = {}
    for path_to_file in to_process:
        if os.path.exists(path_to_file):
            df = pd.read_csv(path_to_file)
            data = df.loc[len(df) - 1, cols]
            model_name = path_to_file.split(os.path.sep)[-2]
            dataset_name = path_to_file.split(os.path.sep)[-3]
            if not structure.get(model_name, None):
                structure[model_name] = []
            values = {}
            for k, v in data.items():
                metrics.add(k)
                avg = v.split("(")[0].strip()
                std = re.search(r'\((.*?)\)', v).group(1)
                values[k] = {"avg": avg, "std": std}
            structure[model_name].append(values)
            datasets.add(dataset_name)
    buf = create_template(
        path_to_tmpl, list(datasets), list(metrics), structure,
        caption=table_caption,
        label=table_label,
        column_positions="l" + "c" * ((len(metrics) * 2 - 1) * len(datasets) + 1)
    )
    with open(out_fname, "w") as f:
        f.write(buf)


def main():
    args = parse_args()
    csv2tex(
        fname=args.results_fname,
        out_fname=args.out,
        path_to_tmpl=args.template,
        from_folders=args.from_folders,
        cols=args.columns,
        table_caption=args.table_caption,
        table_label=args.table_label
    )


if __name__ == "__main__":
    main()
