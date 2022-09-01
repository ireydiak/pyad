import os

from pyad.datamanager.dataset import TabularDataset
from pyad.models.trainer import ModuleTrainer
from pyad.utilities.cli import CLI
from pyad.utilities import instantiate_class


def train(
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


def main(cli):
    args = cli()
    train(
        model_cfg=args.model,
        trainer=args.trainer,
        data=args.data
    )


if __name__ == '__main__':
    main(
        CLI()
    )
