"""Definitions of solvers"""
from abc import abstractmethod, ABC
from typing import Callable

import torch

from sde_lib import SDE


class Solver(ABC):
    """
    The Solver abstract class. The solver takes any SDE and uses a numerical integration technique to approximate the
    solution at a given time.
    """

    def __init__(self, sde: SDE):
        """
        Construct the solver.

        :param sde: The SDE the solver will have to solve.
        """
        self.__sde = sde
        self._device = "cpu"

    @property
    def sde(self):
        return self.__sde

    @abstractmethod
    def solve(self, x: torch.tensor, labels: torch.Tensor = None,
              callback: Callable[[torch.Tensor, torch.Tensor], None] = None) -> torch.tensor:
        """
        Solves the SDE for data x of shape (batch_size, d).

        :param x: Data of shape (batch_size, d).
        :param labels: Optional labels to be passed to the score function
        :param callback: Optional function of (x, t) used for tracking values.
        :return: The data x at time t governed by the SDE.
        """
        pass

    def to(self, device: torch.device | str) -> 'Solver':
        self._device = device
        return self
