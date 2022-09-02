import torch

from typing import Dict, Any


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


def weights_init_xavier(m):
    # Copied from https://github.com/JohnEfan/PyTorch-ALAD/blob/6e7c4a9e9f327b5b08936376f59af2399d10dc9f/utils/utils.py#L4
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        m.bias.data.zero_()
