from typing import Tuple

import numpy as np
import os
import pandas as pd

from datetime import datetime as dt
from torch.utils.data import DataLoader
from tqdm import tqdm
from pyad.datamanager.dataset import TabularDataset
from pyad.models.base import BaseModule
from pyad.utilities import instantiate_class
from pyad.utilities.metrics import AggregatorDict, score_recall_precision_w_threshold, estimate_optimal_threshold


class ModuleTrainer:
    def __init__(
            self,
            max_epochs: int,
            n_runs: int = 1,
            val_check_interval: int = None,
            enable_checkpoints: bool = False,
            enable_early_stopping: bool = False,
            save_dir: str = "results",
            results_fname: str = "results.csv",
            multi_eval_results_fname: str = "multi_evaluation_results.csv"
    ):
        """
        Basic Trainer class to train and test deep neural networks based on the abstract pyad.base.BaseModule class

        Parameters
        ----------
        max_epochs: int
            the number of training epochs

        n_runs: int
            the number of times the experiments are repeated

        val_check_interval: int
            number of epochs between validation steps (***UNIMPLEMENTED***)

        enable_checkpoints: bool
            flag to enable/disable saving checkpoints (***UNIMPLEMENTED***)

        enable_early_stopping: bool
            flag to enable/disable saving early stopping (***UNIMPLEMENTED***)

        save_dir: str
            base directory where model weights and results are stored
        """
        self.max_epochs = max_epochs
        self.n_runs = n_runs
        self.save_dir = save_dir
        self.results_fname = results_fname
        self.multi_eval_results_fname = multi_eval_results_fname
        self.val_check_interval = val_check_interval
        # TODO: enable checkpoints
        self.enable_checkpoints = enable_checkpoints
        # TODO: enable early stopping
        self.enable_early_stopping = enable_early_stopping
        # results placeholders
        self.multi_eval_results, self.results = None, None
        self.optimizer, self.scheduler = None, None

    def get_params(self) -> dict:
        return {
            "max_epochs": self.max_epochs,
            "n_runs": self.n_runs
        }

    def setup_results(self) -> None:
        self.multi_eval_results = {
            "combined_theoretical": AggregatorDict(["Precision", "Recall", "F1-Score", "AUPR", "AUROC", "Thresh"]),
            "combined_optimal": AggregatorDict(["Precision", "Recall", "F1-Score", "AUPR", "AUROC", "Thresh"]),
            "test_only_leaking": AggregatorDict(["Precision", "Recall", "F1-Score", "AUPR", "AUROC", "Thresh"]),
            "test_only_optimal": AggregatorDict(["Precision", "Recall", "F1-Score", "AUPR", "AUROC", "Thresh"]),
            "test_only_theoretical": AggregatorDict(["Precision", "Recall", "F1-Score", "AUPR", "AUROC", "Thresh"]),
        }
        self.results = {}

    def update_results(self, test_train_scores, y_train_test_true, test_scores, y_test_true) -> float:
        # Evaluation
        # set thresholds
        ratio_expected = ((y_train_test_true == 0).sum() / len(y_train_test_true)) * 100
        ratio_test = ((y_test_true == 0).sum() / len(y_test_true)) * 100
        # compute scores
        res, _ = score_recall_precision_w_threshold(
            test_train_scores, y_train_test_true, ratio=ratio_expected
        )
        self.multi_eval_results["combined_theoretical"].add(res)
        res, _ = estimate_optimal_threshold(
            test_train_scores, y_train_test_true, ratio=ratio_expected
        )
        self.multi_eval_results["combined_optimal"].add(res)
        res, _ = score_recall_precision_w_threshold(
            test_scores, y_test_true, ratio=ratio_test
        )
        self.multi_eval_results["test_only_leaking"].add(res)
        res, _ = estimate_optimal_threshold(
            test_scores, y_test_true, ratio=ratio_test
        )
        self.multi_eval_results["test_only_optimal"].add(res)
        f_score = res["F1-Score"]
        res, _ = score_recall_precision_w_threshold(
            test_scores, y_test_true, ratio=ratio_expected
        )
        self.multi_eval_results["test_only_theoretical"].add(res)
        return f_score

    def aggregate_results(self, model_params: dict, data_params: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # aggregate results
        for k, v in self.multi_eval_results.items():
            self.multi_eval_results[k] = v.aggregate()
        self.results = self.multi_eval_results["combined_optimal"]

        # Store Results
        # prepare dataframe with multiple evaluation protocols
        multi_eval_df = pd.DataFrame.from_dict(self.multi_eval_results).T
        agg_results_fname = os.path.join(self.save_dir, self.results_fname)
        # prepare general results dataframe
        row_data = dict(
            **self.results,
            **model_params,
            **self.get_params(),
            **data_params,
        )
        row, cols = prepare_df(row_data)
        results_df = get_or_create_df(agg_results_fname, row, cols, index_col="Timestamp")
        # save dataframes
        return multi_eval_df, results_df

    def _run_once(self, model: BaseModule, train_ldr: DataLoader, validation_ldr: DataLoader = None):
        # TODO: add validation logic after n epochs
        model.on_before_fit(train_ldr)
        for epoch in range(self.max_epochs):
            model.current_epoch = epoch
            model.on_train_epoch_start()
            with tqdm(train_ldr, leave=True) as t_epoch:
                t_epoch.set_description(f"Epoch {epoch + 1}")
                for X, y, full_labels in t_epoch:
                    y = y.to(model.device).float()
                    X = X.to(model.device).float()
                    # clear gradients
                    self.optimizer.zero_grad()
                    # compute forward pass
                    loss = model.training_step(X, y, full_labels)
                    # compute backward pass
                    loss.backward()
                    # update parameters
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    # validation step and log
                    if validation_ldr is not None and self.val_check_interval and (epoch + 1) % self.val_check_interval == 0:
                        # compute score un validation set
                        test_scores, y_test_true, _ = model.predict(validation_ldr)
                        res, _ = score_recall_precision_w_threshold(
                            test_scores, y_test_true
                        )
                        t_epoch.set_postfix(loss='{:.6f}'.format(loss.item()), f_score=res["F1-Score"])
                    else:
                        t_epoch.set_postfix(loss='{:.6f}'.format(loss.item()))
                    t_epoch.update()
            model.on_train_epoch_end()

    def run_experiments(self, model_cfg: dict, data: TabularDataset) -> None:
        # setup
        self.setup_results()
        n_instances = len(data)
        in_features = data.shape[1]
        model = instantiate_class(init=model_cfg, n_instances=n_instances, in_features=in_features)
        model_name = model.print_name()
        dataset_name = data.name
        print("Running %d experiments with model %s on dataset %s" % (self.n_runs, model_name, dataset_name))
        now = dt.now().strftime("%d-%m-%Y_%H-%M-%S")
        # start training
        for run in range(self.n_runs):
            # create fresh model
            model = instantiate_class(init=model_cfg, n_instances=n_instances, in_features=in_features)
            # create optimizer and scheduler
            self.optimizer, self.scheduler = model.configure_optimizers()
            train_ldr, test_ldr = data.loaders(seed=run + 1)
            # fit model on training data
            self._run_once(model, train_ldr=train_ldr, validation_ldr=test_ldr)
            # evaluate model on test set
            f_score = self.test(model, train_ldr, test_ldr)
            print("\nRun {}: f_score={:.4f}".format(run + 1, f_score))
        # aggregate and save results
        multi_eval_df, results_df = self.aggregate_results(model.get_params(), data.get_params())
        agg_results_fname = os.path.join(self.save_dir, self.results_fname)
        multi_eval_save_dir = os.path.join(self.save_dir, now)
        if not os.path.exists(multi_eval_save_dir):
            os.mkdir(multi_eval_save_dir)
        results_df.to_csv(agg_results_fname)
        multi_eval_df.to_csv(
            os.path.join(multi_eval_save_dir, self.multi_eval_results_fname)
        )

    def test(self, model: BaseModule, train_ldr: DataLoader, test_ldr: DataLoader):
        # Predict
        test_scores, y_test_true, _ = model.predict(test_ldr)
        train_scores, y_train_true, _ = model.predict(train_ldr)
        # merge scores
        train_test_scores = np.concatenate((train_scores, test_scores))
        y_true_train_test = np.concatenate((y_train_true, y_test_true))
        f_score = self.update_results(train_test_scores, y_true_train_test, test_scores, y_test_true)
        return f_score


def prepare_df(row_data: dict):
    columns = ["Timestamp"] + list(row_data.keys())
    now = dt.now().strftime("%d/%m/%Y %H:%M:%S")
    row = [now] + list(row_data.values())
    return row, columns


def get_or_create_df(fname: str, row, columns, index_col="Timestamp"):
    df = pd.DataFrame([row], columns=columns)
    if os.path.exists(fname):
        old_df = pd.read_csv(fname)
        df = pd.concat((old_df, df))
    df = df.set_index(index_col)
    return df
