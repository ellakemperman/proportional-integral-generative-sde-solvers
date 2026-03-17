"""Definitions of the stochastic differential equations"""
from abc import abstractmethod
import torch


class SDE:

    def __init__(self):
        pass

    def sde(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._drift(x, t), self._diffusion(t)

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sde(x, t)

    @abstractmethod
    def _drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _diffusion(self, t: torch.Tensor) -> torch.Tensor:
        pass
