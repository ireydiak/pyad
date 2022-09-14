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
    # files to process
    to_process = []
    datasets = []
    for base_root in from_folders:
        assert os.path.exists(base_root), f"path {base_root} does not exist, aborting"
        datasets.append(base_root.split(os.path.sep)[-1])
        to_process.extend(list(
            map(lambda folder: os.path.join(base_root, folder, fname), os.listdir(base_root))
        ))
    metrics, structure = set(), {}
    # will be used a table titles
    datasets = set([ptf.split(os.path.sep)[-3] for ptf in to_process])
    structure = {}
    # read csv files
    for path_to_file in to_process:
        if os.path.exists(path_to_file):
            # load dataframe
            df = pd.read_csv(path_to_file)
            # read last line
            data = df.loc[len(df) - 1, cols]
            model_name = path_to_file.split(os.path.sep)[-2]
            # initialize structure for new models
            if not structure.get(model_name, None):
                structure[model_name] = []
            # add average and standard deviation to structure
            values = {}
            for k, v in data.items():
                metrics.add(k)
                avg = v.split("(")[0].strip()
                std = re.search(r'\((.*?)\)', v).group(1)
                values[k] = {"avg": avg, "std": std}
            structure[model_name].append(values)
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
        fname=args.fname,
        out_fname=args.out,
        path_to_tmpl=args.template,
        from_folders=args.folders,
        cols=args.columns,
        table_caption=args.table_caption,
        table_label=args.table_label
    )


if __name__ == "__main__":
    main()
