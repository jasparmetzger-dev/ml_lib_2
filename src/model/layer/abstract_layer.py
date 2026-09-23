from abc import ABC, abstractmethod

from src.math.tensor import Tensor

class AbstractLayer(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        ...
    @abstractmethod
    def backward(self, X: Tensor) -> Tensor:
        ...
