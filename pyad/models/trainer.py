from typing import Tuple

import numpy as np
import os
import pandas as pd
import torch
import pickle

from datetime import datetime as dt
from torch.utils.data import DataLoader
from tqdm import tqdm
from pyad.datamanager.dataset import TabularDataset
from pyad.loggers import BaseLogger
from pyad.models.base import BaseModule
from pyad.models.shallow import BaseShallowModel
from pyad.utilities import instantiate_class
from pyad.utilities import metrics
from pyad.utilities.cli import TRAINER_REGISTRY
from pyad.utilities.fsystem import mkdir_if_not_exists


@TRAINER_REGISTRY
class ModuleTrainer:
    def __init__(
            self,
            max_epochs: int,
            n_runs: int = 1,
            device: str = "cuda",
            val_check_interval: int = 1,
            enable_checkpoints: bool = False,
            enable_early_stopping: bool = False,
            save_dir: str = "results",
            results_fname: str = "results.csv",
            multi_eval_results_fname: str = "multi_evaluation_results.csv",
            resume_from_checkpoint: str = None,
            checkpoint_interval: int = None,
            checkpoint_fname: str = None,
            logger: BaseLogger = None
    ):
        """
        Basic Trainer class to train and test deep neural networks based on the abstract pyad.base.BaseModule class

        Parameters
        ----------
        max_epochs: int
            the number of training epochs

        n_runs: int
            the number of times the experiments are repeated

        device: str
            device where tensors are stored (use "cuda" for using GPU and "cpu" for the CPU)

        val_check_interval: int
            number of epochs between validation steps (***UNIMPLEMENTED***)

        enable_checkpoints: bool
            flag to enable/disable saving checkpoints (***UNIMPLEMENTED***)

        enable_early_stopping: bool
            flag to enable/disable saving early stopping (***UNIMPLEMENTED***)

        save_dir: str
            base directory where model weights and results are stored

        resume_from_checkpoint: str
            path to checkpoint file

        checkpoint_interval: int
            number of epochs before saving checkpoints

        checkpoint_fname : str
            checkpoint filename
        """
        self.max_epochs = max_epochs
        self.n_runs = n_runs
        self.device = device
        self.save_dir = save_dir
        self.results_fname = results_fname
        self.multi_eval_results_fname = multi_eval_results_fname
        self.val_check_interval = val_check_interval
        self.resume_from_checkpoint = resume_from_checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_fname = checkpoint_fname
        self.now = dt.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.enable_checkpoints = enable_checkpoints
        # TODO: enable early stopping
        self.enable_early_stopping = enable_early_stopping
        # results placeholders
        self.multi_eval_results, self.results = None, None
        self.optimizer, self.scheduler = None, None
        # logger
        self.logger = logger or BaseLogger()

    def get_params(self) -> dict:
        return {
            "max_epochs": self.max_epochs,
            "n_runs": self.n_runs
        }

    def setup_results(self) -> None:
        self.multi_eval_results = {
            "combined_theoretical": metrics.AggregatorDict(),
            "combined_optimal": metrics.AggregatorDict(),
            "test_only_leaking": metrics.AggregatorDict(),
            "test_only_optimal": metrics.AggregatorDict(),
            "test_only_theoretical": metrics.AggregatorDict(),
        }
        self.results = {}

    def update_results(
            self,
            test_train_scores: np.ndarray,
            y_train_test_true: np.ndarray,
            train_labels: np.ndarray,
            test_scores: np.ndarray,
            y_test_true: np.ndarray,
            test_labels: np.ndarray,
            normal_str_repr: str = "0"
    ) -> float:
        # Evaluation
        # set thresholds
        ratio_expected = ((y_train_test_true == 0).sum() / len(y_train_test_true)) * 100
        ratio_test = ((y_test_true == 0).sum() / len(y_test_true)) * 100
        # compute scores
        res, _ = metrics.score_recall_precision_w_threshold(
            test_train_scores, y_train_test_true, ratio=ratio_expected
        )
        self.multi_eval_results["combined_theoretical"].add(res)
        res, _ = metrics.estimate_optimal_threshold(
            test_train_scores, y_train_test_true, ratio=ratio_expected
        )
        self.multi_eval_results["combined_optimal"].add(res)
        res, _ = metrics.score_recall_precision_w_threshold(
            test_scores, y_test_true, ratio=ratio_test
        )
        self.multi_eval_results["test_only_leaking"].add(res)
        res, y_pred = metrics.estimate_optimal_threshold(
            test_scores, y_test_true, ratio=ratio_test
        )
        f_score = res["F1-Score"]
        self.multi_eval_results["test_only_optimal"].add(res)
        res, _ = metrics.score_recall_precision_w_threshold(
            test_scores, y_test_true, ratio=ratio_expected
        )
        self.multi_eval_results["test_only_theoretical"].add(res)
        # compute per-class accuracy if more than two labels in test_labels
        if len(np.unique(test_labels)) > 2:
            pcacc = metrics.per_class_accuracy(y_test_true, y_pred, test_labels, normal_label=normal_str_repr)
            self.multi_eval_results["test_only_optimal"].add(pcacc)
            self.multi_eval_results["combined_optimal"].add(pcacc)
            self.multi_eval_results["test_only_leaking"].add(pcacc)
            self.multi_eval_results["combined_theoretical"].add(pcacc)
        return f_score

    def aggregate_results(self, model_params: dict, data_params: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # aggregate results
        for k, v in self.multi_eval_results.items():
            self.multi_eval_results[k] = v.aggregate()
        self.results = self.multi_eval_results["test_only_optimal"]

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

    def save_model(self, model: BaseModule, run_number: int, epoch: int, path_to_ckpt_file: str) -> None:
        if not path_to_ckpt_file.endswith(".pt"):
            path_to_ckpt_file += ".pt"
        state_dict = {
            "epoch": epoch,
            "run": run_number,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        torch.save(
            state_dict, path_to_ckpt_file
        )

    def load_model(self, model: BaseModule, path_to_ckpt_file: str = None) -> Tuple[BaseModule, int, int]:
        path_to_ckpt_file = path_to_ckpt_file or self.resume_from_checkpoint
        ckpt = torch.load(path_to_ckpt_file)
        model.load_state_dict(
            ckpt["model_state_dict"]
        )
        optimizer, _ = model.configure_optimizers()
        optimizer.load_state_dict(
            ckpt["optimizer_state_dict"]
        )
        self.optimizer = optimizer
        model.train()

        return model, ckpt["run"], ckpt["epoch"]

    def _forward(self, model: BaseModule, X: torch.Tensor, y: torch.Tensor) -> float:
        # clear gradients
        self.optimizer.zero_grad()
        # compute forward pass
        loss = model.training_step(X, y, None)
        # compute backward pass
        loss.backward()
        # update parameters
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()

    def _run_once(
            self,
            model: BaseModule,
            run_number: int,
            train_ldr: DataLoader,
            validation_ldr: DataLoader = None,
            last_epoch: int = 0
    ):
        model.on_before_fit(train_ldr)
        for epoch in range(last_epoch, self.max_epochs):
            model.current_epoch = epoch
            model.on_train_epoch_start()
            with tqdm(train_ldr, leave=True) as t_epoch:
                t_epoch.set_description(f"Epoch {epoch + 1}")
                for X, y, full_labels in t_epoch:
                    y = y.to(model.device).float()
                    X = X.to(model.device).float()
                    loss = self._forward(model, X, y)
                    # validation step and log
                model.on_train_epoch_end()
                if validation_ldr is not None and (epoch + 1) % self.val_check_interval == 0:
                    # compute score un validation set
                    test_scores, y_test_true, _ = model.predict(validation_ldr)
                    res, _ = metrics.score_recall_precision_w_threshold(
                        test_scores, y_test_true
                    )
                    t_epoch.set_postfix(loss='{:.6f}'.format(loss), f_score=res["F1-Score"])
                else:
                    t_epoch.set_postfix(loss='{:.6f}'.format(loss))
                t_epoch.update()
                # save checkpoint
                if self.enable_checkpoints and (epoch + 1) % self.checkpoint_interval == 0:
                    ckpt_path = os.path.join(self.save_dir, "checkpoints", "run_%d" % run_number)
                    ckpt_fname = self.checkpoint_fname or model.print_name() + "_epoch={}.pt".format(epoch)
                    mkdir_if_not_exists(ckpt_path)
                    self.save_model(
                        model, run_number, epoch, os.path.join(ckpt_path, ckpt_fname)
                    )

    def run_experiments(self, model_cfg: dict, data: TabularDataset) -> None:
        # setup
        self.setup_results()
        normal_str_repr = data.normal_str_repr
        n_instances = len(data)
        in_features = data.shape[1]
        model = instantiate_class(
            init=model_cfg, n_instances=n_instances, in_features=in_features, device=self.device
        )
        model_name = model.print_name()
        dataset_name = data.name
        ckpt_model, ckpt_optimizer, last_run, last_epoch = None, None, 0, 0
        # load checkpoint
        if self.resume_from_checkpoint is not None:
            ckpt_model, last_run, last_epoch = self.load_model(model, self.resume_from_checkpoint)
            self.n_runs -= last_run
            self.n_runs = max(self.n_runs, 1)
        # print current configuration
        print("Running %d experiments with model %s on dataset %s using device %s" % (
            self.n_runs, model_name, dataset_name, self.device
        ))
        # log parameters
        self.logger.log("trainer_parameters", self.get_params())
        self.logger.log("parameters", model.get_params())
        self.logger.log_metric("model/", model_name)
        # start training
        for run in range(last_run, self.n_runs):
            # set tags for logger
            self.logger.set_tags(model_name, dataset_name)
            # create fresh model
            if ckpt_model is not None:
                model = ckpt_model
            else:
                model = instantiate_class(
                    init=model_cfg, n_instances=n_instances, in_features=in_features, device=self.device
                )
            # create optimizer and scheduler
            optimizer, self.scheduler = model.configure_optimizers()
            self.optimizer = ckpt_optimizer or optimizer
            # select different normal samples for both training and test sets
            train_ldr, test_ldr = data.loaders()
            # fit model on training data
            self._run_once(model, run, train_ldr=train_ldr, validation_ldr=test_ldr, last_epoch=last_epoch)
            # evaluate model on test set
            f_score = self.test(model, train_ldr, test_ldr, normal_str_repr=normal_str_repr)
            print("\nRun {}: f_score={:.4f}".format(run + 1, f_score))
            ckpt_model, ckpt_optimizer, last_epoch = None, None, 0
            # log results
            for metric, value in self.multi_eval_results["test_only_optimal"].items():
                self.logger.log_metric("eval/" + metric, value[run])
        self.logger.cleanup()

        self.save_results(model, data)
        # save model
        if self.enable_checkpoints:
            model_path = os.path.join("models", dataset_name)
            mkdir_if_not_exists(model_path)
            self.save_model(
                model, run, self.max_epochs,
                os.path.join(model_path, model_name + ".pt")
            )

    def save_results(self, model: BaseModule, data: TabularDataset):
        # aggregate and save results
        multi_eval_df, results_df = self.aggregate_results(model.get_params(), data.get_params())
        agg_results_fname = os.path.join(self.save_dir, self.results_fname)
        multi_eval_save_dir = os.path.join(self.save_dir, self.now)
        mkdir_if_not_exists(multi_eval_save_dir)
        results_df.to_csv(agg_results_fname)
        multi_eval_df.to_csv(
            os.path.join(multi_eval_save_dir, self.multi_eval_results_fname)
        )

    def test(
            self,
            model: BaseModule,
            train_ldr: DataLoader,
            test_ldr: DataLoader,
            normal_str_repr: str = "0"
    ):
        # Predict
        test_scores, y_test_true, test_labels = model.predict(test_ldr)
        train_scores, y_train_true, train_labels = model.predict(train_ldr)
        # merge scores
        train_test_scores = np.concatenate((train_scores, test_scores))
        y_true_train_test = np.concatenate((y_train_true, y_test_true))
        f_score = self.update_results(
            train_test_scores, y_true_train_test, train_labels,
            test_scores, y_test_true, test_labels,
            normal_str_repr=normal_str_repr
        )
        return f_score


@TRAINER_REGISTRY
class AdversarialModuleTrainer(ModuleTrainer):
    def _forward(self, model: BaseModule, X: torch.Tensor, y: torch.Tensor):
        # get optimizers
        g_opt, d_opt = self.optimizer

        g_opt.zero_grad()
        d_opt.zero_grad()

        # compute forward pass
        loss_g, loss_d = model.training_step(X, y)

        loss_d.backward()
        loss_g.backward()

        # Optimize discriminator
        d_opt.step()
        # Optimize generator
        g_opt.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return loss_d.item() + loss_g.item()

    def save_model(self, model: BaseModule, run_number: int, epoch: int, path_to_ckpt_file: str) -> None:
        g_opt, d_opt = self.optimizer
        if not path_to_ckpt_file.endswith(".pt"):
            path_to_ckpt_file += ".pt"
        state_dict = {
            "epoch": epoch,
            "run": run_number,
            "model_state_dict": model.state_dict(),
            "d_optimizer_state_dict": d_opt.state_dict(),
            "g_optimizer_state_dict": g_opt.state_dict()
        }
        torch.save(
            state_dict, path_to_ckpt_file
        )

    def load_model(self, model: BaseModule, path_to_ckpt_file: str = None) -> Tuple[BaseModule, int, int]:
        path_to_ckpt_file = path_to_ckpt_file or self.resume_from_checkpoint
        ckpt = torch.load(path_to_ckpt_file)
        model.load_state_dict(
            ckpt["model_state_dict"]
        )
        optimizers, _ = model.configure_optimizers()
        g_opt, d_opt = optimizers
        g_opt.load_state_dict(
            ckpt["g_optimizer_state_dict"]
        )
        d_opt.load_state_dict(
            ckpt["d_optimizer_state_dict"]
        )
        self.optimizer = (g_opt, d_opt)
        model.train()

        return model, ckpt["run"], ckpt["epoch"]


@TRAINER_REGISTRY
class ShallowModuleTrainer(ModuleTrainer):

    def get_params(self) -> dict:
        return {
            "n_runs": self.n_runs
        }

    def save_model(self, model: BaseShallowModel, epoch: int, run_number: int, path_to_ckpt_file: str) -> None:
        pickle.dump(
            model,
            open(path_to_ckpt_file, "wb")
        )

    def load_model(self, model: BaseShallowModel, path_to_ckpt_file: str = None) -> Tuple[BaseShallowModel, int, int]:
        path_to_ckpt_file = path_to_ckpt_file or self.resume_from_checkpoint
        model = pickle.load(
            open(path_to_ckpt_file, "rb")
        )
        return model, -1, -1

    def run_experiments(self, model_cfg: dict, data: TabularDataset) -> None:
        # setup
        self.setup_results()
        normal_str_repr = data.normal_str_repr
        model = instantiate_class(
            init=model_cfg
        )
        model_name = model.print_name()
        dataset_name = data.name

        # print current configuration
        print("Running %d experiments with model %s on dataset %s" % (
            self.n_runs, model_name, dataset_name
        ))
        if self.resume_from_checkpoint is not None:
            _, last_run, _ = self.load_model(model, self.resume_from_checkpoint)
            self.n_runs -= last_run
            self.n_runs = max(self.n_runs, 1)
        # log parameters
        self.logger.log_metric("model/", model_name)
        self.logger.log("trainer_parameters", self.get_params())
        self.logger.log("parameters", model.get_params())
        # start training
        for run in range(self.n_runs):
            # set tags for logger
            self.logger.set_tags(model_name, dataset_name)
            # create fresh model
            model = instantiate_class(
                init=model_cfg
            )
            # select different normal samples for both training and test sets
            train_ldr, test_ldr = data.loaders()
            # fit shallow model
            model.fit(train_ldr.dataset.X)
            # test
            f_score = self.test(model, train_ldr, test_ldr, normal_str_repr=normal_str_repr)
            if self.enable_checkpoints:
                ckpt_path = os.path.join("results", dataset_name, "run_%d" % run)
                mkdir_if_not_exists(ckpt_path)
                self.save_model(
                    model, -1, run, os.path.join(ckpt_path, model_name + ".sav")
                )
            # log results
            for metric, value in self.multi_eval_results["test_only_optimal"].items():
                self.logger.log_metric("eval/" + metric, value[run])
            print("\nRun {}: f_score={:.4f}".format(run + 1, f_score))
        self.logger.cleanup()
        self.save_results(model, data)
        # save model
        if self.enable_checkpoints:
            model_path = os.path.join("models", dataset_name)
            mkdir_if_not_exists(model_path)
            self.save_model(
                model, -1, self.n_runs,
                os.path.join(model_path, model_name + ".sav")
            )

    def test(
            self,
            model: BaseShallowModel,
            train_ldr: DataLoader,
            test_ldr: DataLoader,
            normal_str_repr: str = "0"
    ):
        train_set = train_ldr.dataset
        test_set = test_ldr.dataset
        # Predict
        test_scores = model.predict(test_set.X)
        y_test_true, test_labels = test_set.y, test_set.labels
        train_scores = model.predict(train_set.X)
        y_train_true, train_labels = train_set.y, train_set.labels
        # merge scores
        train_test_scores = np.concatenate((train_scores, test_scores))
        y_true_train_test = np.concatenate((y_train_true, y_test_true))
        f_score = self.update_results(
            train_test_scores, y_true_train_test, train_labels,
            test_scores, y_test_true, test_labels,
            normal_str_repr=normal_str_repr
        )
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
