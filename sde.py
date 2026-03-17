"""Definitions of the stochastic differential equations"""
from abc import abstractmethod
import torch


class SDE:

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass
