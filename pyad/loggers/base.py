from typing import Any


class BaseLogger:
    def __init__(self, *args, **kwargs):
        pass

    def reinitialize(self) -> None:
        pass

    def log(self, key: Any, value: Any) -> None:
        pass

    def log_metric(self, key: Any, value: Any) -> None:
        pass

    def set_tags(self, *tags) -> None:
        pass

    def cleanup(self) -> None:
        pass
