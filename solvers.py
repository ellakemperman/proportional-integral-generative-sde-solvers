"""Definitions of solvers"""
from itertools import pairwise
from abc import abstractmethod, ABC
from typing import Callable
import torch
from sde import SDE
import time


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

    @property
    def sde(self):
        return self.__sde

    @abstractmethod
    def solve(self, x: torch.tensor, callback: Callable[[torch.Tensor, torch.Tensor], None] = None) -> torch.tensor:
        """
        Solves the SDE for data x of shape (batch_size, d).

        :param x: Data of shape (batch_size, d).
        :param callback: Optional function of (x, t) used for tracking values.
        :return: The data x at time t governed by the SDE.
        """
        pass


class EulerMarayumaSolver(Solver):
    """
    The Euler Marayuma solver uses a simple first order scheme and pre-determined time-discretisation to solve the SDE.
    """

    def __init__(self, sde: SDE, discretisation: torch.Tensor):
        r"""
        Constructs the EulerMarayuma Solver

        :param sde: The SDE the solver will have to solve.
        :param discretisation: The time steps :math:`(t_0, t_1, ..., t_n)` the solver will solve the SDE over.
        """
        super().__init__(sde)

        self._discretisation = discretisation
        self._time_steps = map(lambda x: x[1] - x[0], pairwise(discretisation))

    def solve(self, x: torch.tensor, callback: Callable[[torch.Tensor, torch.Tensor], None] = None) -> torch.tensor:
        for i, dt in enumerate(self._time_steps):
            t = self._discretisation[i]
            x = self.step(x, t, dt)
            if callback is not None:
                callback(x, t + dt)

        return x

    def step(self, x: torch.tensor, t: torch.tensor, dt: torch.tensor) -> torch.tensor:
        """
        Computes one Euler-Marayuma step on the data x.

        :param x: Data of shape (batch_size, d).
        :param t: Scalar time
        :param dt: Scalar time-step
        :return: x + dx
        """
        return x + self.sde.step(x, t, dt)


class PISolver(Solver):
    """
    The PISolver uses an adaptive proportional-integral method to solve the SDE. It dynamically adjusts the step size
    based on local measured error for each sample in the batch. This error is computed using an embedded stochastic
    Runge-Kutta method of second order, extrapolating this and comparing it with the first-order error.
    """

    def __init__(self,
                 sde: SDE,
                 ki: float,
                 kp: float,
                 tau: float,
                 alpha: float,
                 h_start: float,
                 max_increase: float,
                 max_decrease: float,
                 interval: tuple[float, float] = (1, 0),
                 timeout: float = 20
                 ):
        r"""
        Constructs the PISolver.

        :param sde: The SDE the solver will have to solve.
        :param ki: The integral constant, determining influence of integral component.
        :param kp: The proportional constant, determining influence of integral component.
        :param tau: The tolerance tau, higher meaning the solver has a higher tolerance for error.
        :param alpha: The safety factor, for which :math:`\alpha \leq 1`, reduces chance of rejecting next step size.
        :param h_start: The starting step size.
        :param max_increase: A factor determining the maximum increase of the step size for each step.
        :param max_decrease: A factor determining the maximum decrease of the step size for each step.
        :param interval: The interval over which the SDE is computed
        """
        super().__init__(sde)

        self._ki = ki
        self._kp = kp
        self._tau = tau
        self._alpha = alpha
        self._start_time = float(interval[0])
        self._end_time = float(interval[1])
        self._h_start = abs(h_start) * (self._end_time - self._start_time) / abs(self._end_time - self._start_time)
        self._max_increase = torch.Tensor([max_increase])
        self._max_decrease = torch.Tensor([max_decrease])
        self._timeout = timeout

    def solve(self, x: torch.Tensor, callback: Callable[[torch.Tensor, torch.Tensor], None] = None) -> torch.Tensor:
        t = torch.full((x.shape[0], 1), self._start_time)  # Initialise batch_size times, starting at 1
        h = torch.full((x.shape[0], 1), self._h_start)
        error = torch.full((x.shape[0], 1), 0.5)
        end_condition = torch.full((x.shape[0], 1), self._end_time)
        reject_count = 0
        not_reject_count = 0
        x_full, t_full, h_full = x, t, h
        start_time = time.time()

        # Loop until all times in the batch are equal to the end time
        while torch.any(not_finished := (t_full != end_condition)):
            # Work with the unfinished subset of x, t, h
            not_finished = not_finished.reshape(-1)
            x, t, h = x_full[not_finished], t_full[not_finished], h_full[not_finished]

            # Perform Euler and Heun step
            w = torch.randn_like(x)
            dx_euler = self.sde.step(x, t, h, w)
            dx_heun = self.sde.step(x + dx_euler, t + h, h, w)

            # Compute first and second order x
            x_first = x + dx_euler
            x_second = x + 0.5 * (dx_euler + dx_heun)

            # Compute extrapolated error
            old_error = error.clone()
            error[not_finished] = self._error(x_first, x_second)

            # Update x and t
            not_rejected = error[not_finished] < 1
            x[not_rejected] = x_second[not_rejected]
            t[not_rejected] = t[not_rejected] + h[not_rejected]

            reject_count += torch.sum(error > 1)
            not_reject_count += torch.sum(not_rejected)

            # Get next step size
            h = self._get_next_step(error[not_finished], old_error[not_finished], h)

            # Bound steps such that no step exceeding end condition will be taken
            h = torch.maximum(h, end_condition[not_finished] - t)

            # Update the full matrices
            x_full[not_finished], t_full[not_finished], h_full[not_finished] = x, t, h

            # Timeout
            if time.time() >= start_time + self._timeout:
                raise TimeoutError

            if callback is not None:
                callback(x_full, t_full)

        print(reject_count / (reject_count + not_reject_count))
        return x_full

    def _error(self, x_first: torch.Tensor, x_second: torch.Tensor) -> torch.Tensor:
        """
        Computes the local extrapolated error by computing the MSE divided by the tolerance.

        :param x_first: The first order estimate of x, tensor of shape (batch_size, d).
        :param x_second: The higher order estimate of x, tensor of shape (batch_size, d).
        :return: The error, a tensor of shape (batch_size, 1)
        """
        d = sum(x_first.shape[1:])
        return torch.sqrt(torch.sum(torch.square((x_second - x_first) / self._tau), dim=1) / d).unsqueeze(-1)

    def _get_next_step(self, error: torch.Tensor, error_previous, h: torch.Tensor) -> torch.Tensor:
        """
        Computes the next step h for each x based on the error using a PI controller.

        :param error: Current error, tensor of shape (batch_size, 1).
        :param error_previous: Error of previous step, tensor of shape (batch_size, 1).
        :param h: Current step size, tensor of shape (batch_size, 1).
        :return: The new step size, a tensor of shape (batch_size, 1).
        """
        integral = (self._alpha * self._tau / error)**(self._ki + self._kp)
        proportional = (error_previous / (self._alpha * self._tau))**self._kp
        return h * torch.min(self._max_increase, torch.max(self._max_decrease, integral * proportional))


class PredictorCorrectorSolver(Solver):

    def __init__(self, sde: SDE):
        super().__init__(sde)
