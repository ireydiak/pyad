import inspect

from typing import Type, Optional, Generator, List, Tuple, Any, Dict
from types import ModuleType
from jsonargparse import ActionConfigFile, ArgumentParser
from pyad.datamanager.dataset import TabularDataset


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


def instantiate_class(init: Dict[str, Any], **kwargs) -> Any:
    """Instantiates a class with the given args and init.
    *** FUNCTION COPIED FROM the Lightning library (https://www.pytorchlightning.ai/) ***
    Args:
        args: Positional arguments required for instantiation.
        init: Dict of the form {"class_path":...,"init_args":...}.

    Returns:
        The instantiated class object.
    """
    init_args = init.get("init_args", {})
    init_args = dict(**init_args, **kwargs)
    for key, item in init_args.items():
        if type(item) is dict and "class_path" in set(item.keys()):
            init_args[key] = instantiate_class(item, **kwargs)
    class_module, class_name = init["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(**init_args)


class CLI:
    def __init__(self):
        self.parser = ArgumentParser()
        self.cfg = None
        self.parser.add_argument("--model", type=dict)
        self.parser.add_class_arguments(TabularDataset, "data.init_args")
        self.parser.add_argument("--n_runs", type=int, default=1, help="number of times experiments are repeated")
        self.parser.add_argument("--config", action=ActionConfigFile)
        self.parser.add_argument("--save_dir", type=str, default=None, help="path where experiments files are saved")

    def __call__(self, *args, **kwargs):
        self.cfg = self.parser.parse_args()
        self.cfg = self.parser.instantiate_classes(self.cfg)
        model = MODEL_REGISTRY.get(self.cfg.model["class_path"])
        if model is None:
            raise NotImplementedError("model %s unimplemented" % self.cfg.model["class_path"])
        data = self.cfg.data.init_args
        self.cfg.data = data
        return self.cfg
