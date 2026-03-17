from abc import abstractmethod
from sde import SDE
import torch
from itertools import pairwise


class Solver:

    def __init__(self, sde: SDE):
        self.__sde = sde

    @property
    def sde(self):
        return self.__sde

    @abstractmethod
    def solve(self, x: torch.tensor) -> torch.tensor:
        pass

    def step(self, x: torch.tensor, t: torch.tensor, dt: torch.tensor) -> torch.tensor:
        """
        Computes an SDE step

        :param x:
        :param t:
        :param dt:
        :return:
        """
        drift, diffusion = self.sde(x, t)



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


