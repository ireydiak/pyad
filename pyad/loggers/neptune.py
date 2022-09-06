import neptune.new as neptune
import os

from typing import Any
from pyad.loggers.base import BaseLogger


class NeptuneLogger(BaseLogger):

    def __init__(self, tags=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tags = tags or []
        self.neptune = neptune.init(
            api_token=os.environ["NEPTUNE_API_TOKEN"],
            project=os.environ["NEPTUNE_PROJECT"],
            tags=tags,
            name="test"
        )

    def log(self, key: Any, value: Any) -> None:
        self.neptune[key] = value

    def log_metric(self, key: Any, value: Any) -> None:
        self.neptune[key].log(value)

    def set_tags(self, *tags) -> None:
        self.tags = tags
        self.neptune["sys/tags"].add(self.tags)

    def reinitialize(self) -> None:
        self.neptune = neptune.init(
            api_token=os.environ["NEPTUNE_API_TOKEN"],
            project=os.environ["NEPTUNE_PROJECT"],
            tags=self.tags
        )

    def cleanup(self) -> None:
        self.neptune.stop()
