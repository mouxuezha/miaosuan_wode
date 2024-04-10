from abc import ABC, abstractmethod
from typing import List


class BaseAgent(ABC):
    @abstractmethod
    def setup(self, setup_info: dict) -> None:
        pass

    @abstractmethod
    def step(self, observation: dict) -> List[dict]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
