import inspect

from typing import Type, Optional, Generator, List, Tuple, Any, Dict
from types import ModuleType

import yaml
from jsonargparse import ActionConfigFile, ArgumentParser
from pyad.datamanager.dataset import TabularDataset
from pyad.utilities import instantiate_class


def merge_dict(a: dict, b: dict) -> dict:
    new_dict = a.copy()
    for k, v in b.items():
        new_dict[k] = v
    return new_dict


def merge_configs(configs: List[str], key: str):
    final_cfg = {}
    for f in configs:
        with open(f, "r") as stream:
            f_cfg = yaml.safe_load(stream)
            if f_cfg.get(key, None):
                for k, v in f_cfg.get(key).items():
                    if type(v) == dict and final_cfg.get(k, None):
                        v = merge_dict(final_cfg[k], v)
                    final_cfg[k] = v
    return final_cfg


class _Registry(dict):
    """
        *** CLASS COPIED FROM the Lightning library (https://www.pytorchlightning.ai/) ***
    """

    def __call__(self, cls: Type, key: Optional[str] = None, override: bool = False) -> Type:
        """Registers a class mapped to a name.

        Args:
            cls: the class to be mapped.
            key: the name that identifies the provided class.
            override: Whether to override an existing key.
        """
        if key is None:
            key = cls.__module__ + "." + cls.__name__  # .__name__
        elif not isinstance(key, str):
            raise TypeError(f"`key` must be a str, found {key}")

        if key not in self or override:
            self[key] = cls
        return cls

    def register_classes(self, module: ModuleType, base_cls: Type, override: bool = False) -> None:
        """This function is a utility to register all classes from a module."""
        for cls in self.get_members(module, base_cls):
            self(cls=cls, override=override)

    @staticmethod
    def get_members(module: ModuleType, base_cls: Type) -> Generator[Type, None, None]:
        return (
            cls
            for _, cls in inspect.getmembers(module, predicate=inspect.isclass)
            if issubclass(cls, base_cls) and cls != base_cls
        )

    @property
    def names(self) -> List[str]:
        """Returns the registered names."""
        return list(self.keys())

    @property
    def classes(self) -> Tuple[Type, ...]:
        """Returns the registered classes."""
        return tuple(self.values())

    def __str__(self) -> str:
        return f"Registered objects: {self.names}"


MODEL_REGISTRY = _Registry()
DATAMODULE_REGISTRY = _Registry()
TRAINER_REGISTRY = _Registry()


class CLI:
    def __init__(self):
        self.parser = ArgumentParser()
        self.cfg = None
        self.parser.add_argument("method", type=str, default="fit")
        self.parser.add_argument("--model", type=dict)
        self.parser.add_argument("--trainer", type=dict)
        self.parser.add_argument("--debug", action="store_true")
        self.parser.add_class_arguments(TabularDataset, "data.init_args")
        self.parser.add_argument("--config", action=ActionConfigFile)

    def __call__(self, *args, **kwargs):
        self.cfg = self.parser.parse_args()
        self.cfg = self.parser.instantiate_classes(self.cfg)
        trainer_cfg = merge_configs(list(map(lambda f: f.abs_path, self.cfg.config)), "trainer")
        model = MODEL_REGISTRY.get(self.cfg.model["class_path"])
        trainer = TRAINER_REGISTRY.get(trainer_cfg["class_path"])
        if model is None:
            raise NotImplementedError("model %s unimplemented" % self.cfg.model["class_path"])
        if trainer is None:
            raise NotImplementedError("trainer %s unimplemented" % self.cfg.trainer["class_path"])
        data = self.cfg.data.init_args
        self.cfg.data = data
        self.cfg.trainer = instantiate_class(init=trainer_cfg)
        return self.cfg
