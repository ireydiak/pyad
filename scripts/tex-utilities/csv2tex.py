import os
import re

import pandas as pd
import jinja2
from jinja2 import Template


def create_template(path_to_template, datasets, metrics, structure, caption: str, label: str, column_positions: str) -> str:
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


def main():
    fname = "results.csv"
    out_fname = "output.tex"
    path_to_template = "examples/table_tmpl.tex"
    results_base_roots = [
        "C:\\Users\\verj2009\\NRCAN\\sandbox\\pyad\\results\\arrhythmia",
        "C:\\Users\\verj2009\\NRCAN\\sandbox\\pyad\\results\\arrhythmia_backup"
    ]
    to_process = []
    for base_root in results_base_roots:
        to_process.extend(list(
            map(lambda folder: os.path.join(base_root, folder, fname), os.listdir(base_root))
        ))
    cols = ["AUPR", "AUROC"]
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
        path_to_template, list(datasets), list(metrics), structure,
        caption="Average precision, recall, and f1-score with standard deviation of the baseline models trained on different intrusion detection and medical datasets. For each metric, the best result is shown in bold.",
        label="tab:threshold-results",
        column_positions="l" + "c" * ((len(metrics) * 2 - 1) * len(datasets) + 1)
    )
    with open(out_fname, "w") as f:
        f.write(buf)


if __name__ == "__main__":
    main()
