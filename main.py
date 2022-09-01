import numpy as np
import pandas as pd
import os

from pyad.models.base import BaseModule
from pyad.datamanager.dataset import TabularDataset
from pyad.utilities.metrics import AggregatorDict, score_recall_precision_w_threshold, estimate_optimal_threshold
from pyad.utilities.cli import CLI, instantiate_class
from datetime import datetime as dt


def train(
        model_cfg: dict,
        data: TabularDataset,
        n_runs: int,
        save_dir: str = None
):
    dataset_name = data.name.lower()
    model = instantiate_class(init=model_cfg, n_instances=data.n_instances, in_features=data.in_features)
    model_name = model.print_name()

    save_dir = save_dir or os.path.join("results", dataset_name, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    multi_eval_results = {
        "combined_theoretical": AggregatorDict(["Precision", "Recall", "F1-Score", "AUPR", "AUROC", "Thresh"]),
        "combined_optimal": AggregatorDict(["Precision", "Recall", "F1-Score", "AUPR", "AUROC", "Thresh"]),
        "test_only_leaking": AggregatorDict(["Precision", "Recall", "F1-Score", "AUPR", "AUROC", "Thresh"]),
        "test_only_optimal": AggregatorDict(["Precision", "Recall", "F1-Score", "AUPR", "AUROC", "Thresh"]),
        "test_only_theoretical": AggregatorDict(["Precision", "Recall", "F1-Score", "AUPR", "AUROC", "Thresh"]),
    }
    now = dt.now().strftime("%d-%m-%Y_%H-%M-%S")

    print("Running %d experiments with model %s on dataset %s" % (n_runs, model_name, dataset_name))
    for run in range(n_runs):
        # Generate data loaders
        train_ldr, test_ldr = data.loaders(seed=run + 1)
        # Start training process
        model = instantiate_class(init=model_cfg, n_instances=data.n_instances, in_features=data.in_features)
        model.fit(train_ldr)
        # Predict
        test_scores, y_test_true, _ = model.predict(test_ldr)
        train_scores, y_train_true, _ = model.predict(train_ldr)
        # merge scores
        combined_scores = np.concatenate((train_scores, test_scores))
        combined_y = np.concatenate((y_train_true, y_test_true))

        # Evaluation
        # set thresholds
        ratio_expected = (1 - data.anomaly_ratio) * 100
        ratio_test = ((y_test_true == 0).sum() / len(y_test_true)) * 100
        # compute scores
        res, _ = score_recall_precision_w_threshold(
            combined_scores, combined_y, ratio=ratio_expected
        )
        multi_eval_results["combined_theoretical"].add(res)
        res, _ = estimate_optimal_threshold(
            combined_scores, combined_y, ratio=ratio_expected
        )
        multi_eval_results["combined_optimal"].add(res)
        res, _ = score_recall_precision_w_threshold(
            test_scores, y_test_true, ratio=ratio_test
        )
        multi_eval_results["test_only_leaking"].add(res)
        res, _ = estimate_optimal_threshold(
            test_scores, y_test_true, ratio=ratio_test
        )
        multi_eval_results["test_only_optimal"].add(res)
        f_score = res["F1-Score"]
        res, _ = score_recall_precision_w_threshold(
            test_scores, y_test_true, ratio=ratio_expected
        )
        multi_eval_results["test_only_theoretical"].add(res)
        print("\nRun {}: f_score={:.4f}".format(run + 1, f_score))

    # aggregate results
    for k, v in multi_eval_results.items():
        multi_eval_results[k] = v.aggregate()
    results = multi_eval_results["combined_optimal"]

    # Store Results
    # prepare dataframe with multiple evaluation protocols
    multi_eval_df = pd.DataFrame.from_dict(multi_eval_results).T
    agg_results_fname = os.path.join(save_dir, "results.csv")
    multi_eval_save_dir = os.path.join(save_dir, now)
    os.mkdir(multi_eval_save_dir)
    # prepare general results dataframe
    row, cols = prepare_df(results, model, data, {"n_runs": n_runs})
    df = get_or_create_df(agg_results_fname, row, cols, index_col="Timestamp")
    # save dataframes
    df.to_csv(agg_results_fname)
    multi_eval_df.to_csv(
        os.path.join(multi_eval_save_dir, "multi_evaluation_results.csv")
    )


def prepare_df(results: dict, model: BaseModule, data, other_params: dict = None):
    other_params = other_params or {}
    model_params = model.get_params()
    data_params = data.get_params()
    columns = ["Timestamp"] + list(results.keys()) + list(model_params.keys()) + list(other_params.keys()) + list(data_params.keys())
    now = dt.now().strftime("%d/%m/%Y %H:%M:%S")
    row = [now] + list(results.values()) + list(model_params.values()) + list(other_params.values()) + list(data_params.values())
    return row, columns


def get_or_create_df(fname: str, row, columns, index_col="Timestamp"):
    df = pd.DataFrame([row], columns=columns)
    if os.path.exists(fname):
        old_df = pd.read_csv(fname)
        df = pd.concat((old_df, df))
    df = df.set_index(index_col)
    return df


def main(cli):
    args = cli()
    train(
        model_cfg=args.model,
        data=args.data,
        n_runs=args.n_runs,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main(
        CLI()
    )
