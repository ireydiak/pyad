import os


def mkdir_if_not_exists(path: str, exists_ok: bool = True):
    if not os.path.exists(path):
        os.makedirs(path, exists_ok)
