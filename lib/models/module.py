"""
Generic module class to hold diarization functionality.
"""

from abc import ABC, abstractmethod
from typing import Any 
from logging import getLogger


logger = getLogger(__name__)


class DiarizationModule(ABC):
    def __init__(self, tag: str):
        self.tag = tag

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        logger.info(f"Running module: {self.tag}")
        return self.run(*args, **kwargs)

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Implement in child classes with custom params."""
        raise NotImplementedError 
