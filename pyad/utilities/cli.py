import inspect

from typing import Type, Optional, Generator, List, Tuple, Any, Dict
from types import ModuleType
from jsonargparse import ActionConfigFile, ArgumentParser
from pyad.datamanager.dataset import TabularDataset
from pyad.utilities import instantiate_class


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
        self.parser.add_class_arguments(TabularDataset, "data.init_args")
        self.parser.add_argument("--config", action=ActionConfigFile)

    def __call__(self, *args, **kwargs):
        self.cfg = self.parser.parse_args()
        self.cfg = self.parser.instantiate_classes(self.cfg)
        model = MODEL_REGISTRY.get(self.cfg.model["class_path"])
        trainer = TRAINER_REGISTRY.get(self.cfg.trainer["class_path"])
        if model is None:
            raise NotImplementedError("model %s unimplemented" % self.cfg.model["class_path"])
        if trainer is None:
            raise NotImplementedError("trainer %s unimplemented" % self.cfg.trainer["class_path"])
        data = self.cfg.data.init_args
        self.cfg.data = data
        self.cfg.trainer = instantiate_class(init=self.cfg.trainer)
        return self.cfg
