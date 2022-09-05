import os
import warnings

from pyad.datamanager.dataset import TabularDataset
from pyad.models.trainer import ModuleTrainer
from pyad.utilities.cli import CLI
from pyad.utilities import instantiate_class


def fit(
        model_cfg: dict,
        trainer: ModuleTrainer,
        data: TabularDataset
):
    # setup
    dataset_name = data.name.lower()
    model = instantiate_class(init=model_cfg, n_instances=data.n_instances, in_features=data.in_features)
    model_name = model.print_name()
    save_dir = os.path.join("results", dataset_name, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # update trainer
    trainer.save_dir = save_dir
    # start training
    trainer.run_experiments(model_cfg, data)


def test(
        model_cfg: dict,
        trainer: ModuleTrainer,
        data: TabularDataset
):
    if trainer.enable_checkpoints:
        warnings.warn(
            "`enable_checkpoints` is set to True but in test mode no checkpoints are saved. "
            "To disable this warning set `enable_checkpoints` to False"
        )
    assert trainer.resume_from_checkpoint, "`test` requires parameter `Trainer.resume_from_checkpoint` to load a model"
    # setup
    model = instantiate_class(
        init=model_cfg, n_instances=data.n_instances, in_features=data.in_features
    )
    save_dir = os.path.join("results", data.name, model.print_name())
    # start testing
    model, _, _ = trainer.load_model(model)
    train_ldr, test_ldr = data.loaders()
    trainer.save_dir = save_dir
    trainer.setup_results()
    trainer.enable_checkpoints = False
    f1 = trainer.test(model, train_ldr, test_ldr, normal_str_repr=data.normal_str_repr)
    print("f1-score={:.4f}".format(f1))
    trainer.save_results(model, data)


def main(cli):
    args = cli()
    if args.method == "fit":
        fit(
            model_cfg=args.model,
            trainer=args.trainer,
            data=args.data
        )
    elif args.method == "test":
        test(
            model_cfg=args.model,
            trainer=args.trainer,
            data=args.data
        )


if __name__ == '__main__':
    main(
        CLI()
    )
