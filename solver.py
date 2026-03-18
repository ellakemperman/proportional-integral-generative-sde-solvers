"""Definitions of solvers"""
from itertools import pairwise
from abc import abstractmethod, ABC
import torch
from sde import SDE


class Solver(ABC):

    def __init__(self, sde: SDE):
        self.__sde = sde

    @property
    def sde(self):
        return self.__sde

    @abstractmethod
    def solve(self, x: torch.tensor) -> torch.tensor:
        pass

    @abstractmethod
    def step(self, x: torch.tensor, t: torch.tensor, dt: torch.tensor) -> torch.tensor:
        pass


class EulerMarayumaSolver(Solver):

    def __init__(self, sde: SDE, discretisation: torch.tensor):
        super().__init__(sde)

        self._discretisation = discretisation
        self._time_steps = map(lambda x: x[1] - x[0], pairwise(discretisation))

    def solve(self, x: torch.tensor) -> torch.tensor:
        for i, dt in enumerate(self._time_steps):
            t = self._discretisation[i]
            x = self.step(x, t, dt)

        return x

    def step(self, x: torch.tensor, t: torch.tensor, dt: torch.tensor) -> torch.tensor:
        noise = torch.randn_like(x)
        return self.sde.step(x, t, dt, noise)
