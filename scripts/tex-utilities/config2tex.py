import os
import yaml
import jsonargparse
import jinja2
from typing import List
from argparse import Namespace


def parse_args() -> Namespace:
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--config_root", type=str)
    parser.add_argument("--template", type=str, help="path to latex template")
    parser.add_argument("--table_caption", type=str, default="")
    parser.add_argument("--folders", type=List[str], default=None)
    parser.add_argument("--models", type=List[str], default=None)

    return parser.parse_args()


def config2tex(
        path_to_template: str,
        model_params: List[str],
        table_caption: str,
        column_positions: str,
        structure: dict
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
        params=model_params,
        caption=table_caption,
        column_positions=column_positions,
        structure=structure
    )
    return buf


def get_init_args(
        config_folders: List[str],
        data_fname: str,
        key: str,
        include: List[str] = None,
        exclude: List[str] = None
) -> dict:
    cfg = {}
    for path in config_folders:
        dataset_name = path.split(os.path.sep)[-1]
        cfg[dataset_name] = {}
        path_to_params = os.path.join(path, data_fname)
        with open(path_to_params, "r") as f:
            params = yaml.safe_load(f)
            params = params[key]["init_args"]
            if include is not None:
                cfg[dataset_name] = {k: params.get(k) for k in include}
            elif exclude is not None:
                to_include = list(set(params) - set(exclude))
                cfg[dataset_name] = {k: params.get(k) for k in to_include}
            else:
                cfg[dataset_name] = params
    return cfg


def merge_configs(*args) -> dict:
    final_cfg = {}
    for cfg in args:
        for k, v in cfg.items():
            if not final_cfg.get(k, None):
                final_cfg[k] = v
            else:
                for _k, _v in v.items():
                    final_cfg[k][_k] = _v
    return final_cfg


def resolve_models(config_folders: List[str]) -> List[str]:
    models = set()
    for folder in config_folders:
        for f in os.listdir(folder):
            if not f.startswith("_"):
                models.add(f.replace(".yaml", ""))
    return list(models)


def main():
    args = parse_args()
    if args.folders:
        config_folders = list(map(lambda x: os.path.join(args.config_root, x), args.folders))
    else:
        config_folders = list(map(lambda x: os.path.join(args.config_root, x), os.listdir(args.config_root)))
    models = args.models or resolve_models(config_folders)
    for model_name in models:
        trainer_params = get_init_args(config_folders, "_trainer.yaml", "trainer", include=["max_epochs"])
        data_params = get_init_args(config_folders, "_data.yaml", "data", include=["batch_size", "scaler"])
        model_params = get_init_args(config_folders, "%s.yaml" % model_name, "model", exclude=["trainer", "data"])
        structure = merge_configs(trainer_params, data_params, model_params)
        table_caption = "%s hyperparameters" % model_name
        param_keys = ["Dataset"]
        for d in [trainer_params, data_params, model_params]:
            param_keys.extend(list(
                x.replace("_", " ") for x in d[next(iter(d))].keys()
            ))
        # put first letter uppercase
        param_keys = [s[0].upper() + s[1:] for s in param_keys]

        buf = config2tex(
            path_to_template=args.template,
            model_params=param_keys,
            table_caption=table_caption,
            column_positions="l" + "c" * (len(param_keys) - 1),
            structure=structure
        )
        out_fname = os.path.join("output", "%s_parameters.tex" % model_name)
        with open(os.path.join(out_fname), "w") as f:
            f.write(buf)


if __name__ == "__main__":
    main()
