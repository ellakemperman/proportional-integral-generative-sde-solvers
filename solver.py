"""Definitions of solvers"""
from itertools import pairwise
from abc import abstractmethod, ABC
import torch
from sde import SDE, LinearDriftSDE


class Solver(ABC):

    def __init__(self, sde: SDE):
        self.__sde = sde

    @property
    def sde(self):
        return self.__sde

    @abstractmethod
    def solve(self, x: torch.tensor) -> torch.tensor:
        pass


class EulerMarayumaSolver(Solver):

    def __init__(self, sde: SDE, discretisation: torch.Tensor):
        super().__init__(sde)

        self._discretisation = discretisation
        self._time_steps = map(lambda x: x[1] - x[0], pairwise(discretisation))

    def solve(self, x: torch.tensor) -> torch.tensor:
        for i, dt in enumerate(self._time_steps):
            t = self._discretisation[i]
            x = self.step(x, t, dt)

        return x

    def step(self, x: torch.tensor, t: torch.tensor, dt: torch.tensor) -> torch.tensor:
        return x + self.sde.step(x, t, dt)


class PISolver(Solver):

        def __init__(self,
                     sde: SDE,
                     ki: float,
                     kp: float,
                     tau: float,
                     alpha: float,
                     h_start: float,
                     max_increase: float,
                     max_decrease: float,
                     end_time: float = 0,
                     ):
            super().__init__(sde)

            self._ki = ki
            self._kp = kp
            self._tau = tau
            self._alpha = alpha
            self._h_start = h_start
            self._end_time = end_time
            self._max_increase = torch.Tensor([max_increase])
            self._max_decrease = torch.Tensor([max_decrease])

        def solve(self, x: torch.tensor) -> torch.tensor:
            t = torch.ones(x.shape[0], 1)  # Initialise batch_size times, starting at 1
            h = torch.full((x.shape[0], 1), self._h_start)
            error = torch.full((x.shape[0], 1), 0.5)
            end_condition = torch.full((x.shape[0], 1), self._end_time)
            i = 0
            reject_count = 0
            not_reject_count = 0

            while torch.any(not_finished := (t != end_condition)):
                w = torch.randn_like(x)
                dx_euler = self.sde.step(x, t, h, w)
                dx_heun = self.sde.step(x + dx_euler, t + h, h, w)

                x_first = x + dx_euler
                x_second = x + 0.5 * (dx_euler + dx_heun)

                old_error = error
                error = self._error(x_first, x_second)

                # Update x and t
                not_rejected = error < 1
                x[not_rejected] = x_second[not_rejected]
                t[not_rejected] = t[not_rejected] + h[not_rejected]

                reject_count += torch.sum(error > 1)
                not_reject_count += torch.sum(not_rejected)

                # Get next step size
                h = self._get_next_step(error, old_error, h)

                # Bound steps such that no step exceeding end condition will be taken
                h = torch.maximum(h, end_condition - t)

                i += 1

            print(reject_count / (reject_count + not_reject_count))
            print(i)
            return x

        def _error(self, x_first: torch.Tensor, x_second: torch.Tensor):
            d = sum(x_first.shape[1:])
            return torch.sqrt(torch.sum(torch.square((x_second - x_first) / self._tau), dim=1) / d).unsqueeze(-1)

        def _get_next_step(self, error: torch.Tensor, error_previous, h: torch.Tensor) -> torch.Tensor:
            integral = (self._alpha * self._tau / error)**(self._ki + self._kp)
            proportional = (error_previous / (self._alpha * self._tau))**self._kp
            # Potentially bound h between two factors
            return h * torch.min(self._max_increase, torch.max(self._max_decrease, integral * proportional))
