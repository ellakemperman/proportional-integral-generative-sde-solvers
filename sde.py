"""Definitions of the stochastic differential equations"""
from abc import abstractmethod
import torch
from typing import Callable


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

    def get_reverse_sde(self, score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> 'SDE':
        parent = self

        class ReverseSDE(parent.__class__):

            def __init__(self):
                super().__init__()

            def _drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                return parent._drift(x, t) - torch.square(parent._diffusion(t)) * score_fn(x, t)

            def _diffusion(self, t: torch.Tensor) -> torch.Tensor:
                return parent._diffusion(t)

        return ReverseSDE()
