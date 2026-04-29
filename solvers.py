"""Definitions of solvers"""
from itertools import pairwise
from abc import abstractmethod, ABC
from typing import Callable
import time
import torch
from sde import SDE
from utils import broadcast_vector


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
        self._time_steps = discretisation[1:] - discretisation[:-1]

    def solve(self, x: torch.tensor, labels: torch.Tensor = None,
              callback: Callable[[torch.Tensor, torch.Tensor], None] = None) -> torch.tensor:
        for i, dt in enumerate(self._time_steps):
            t = broadcast_vector((self._discretisation[i] * torch.ones(x.shape[0])).to(self._device), x)
            x += self.sde.step(x, t, dt, labels=labels)

            if callback is not None:
                callback(x, t + dt)

        return x

    def to(self, device: str) -> Solver:
        super().to(device)
        self._discretisation.to(self._device)
        return self


class HeunSolver(EulerMarayumaSolver):

    def solve(self, x: torch.tensor, labels: torch.Tensor = None,
              callback: Callable[[torch.Tensor, torch.Tensor], None] = None) -> torch.tensor:
        for i, dt in enumerate(self._time_steps):
            t = broadcast_vector((self._discretisation[i] * torch.ones(x.shape[0])).to(self._device), x)
            dx_euler = self.sde.step(x, t, dt, labels=labels)

            if i < (self._time_steps.shape[0] - 1):
                dx_heun = self.sde.step(x + dx_euler, t + dt, self._time_steps[i + 1], labels=labels)
                x += 0.5 * (dx_euler + dx_heun)
            else:
                x += dx_euler

            if callback is not None:
                callback(x, t + dt)

        return x


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
                 tau_a: float,
                 tau_r: float,
                 alpha: float,
                 h_start: float,
                 max_increase: float,
                 max_decrease: float,
                 interval: tuple[float, float] = (1, 0),
                 max_iter: int = 10000
                 ):
        r"""
        Constructs the PISolver.

        :param sde: The SDE the solver will have to solve.
        :param ki: The integral constant, determining influence of integral component.
        :param kp: The proportional constant, determining influence of integral component.
        :param tau_a: The absolute tolerance tau_a, the minimum tolerance of the solver.
        :param tau_r: The relative tolerance tau_r, the factor that scales tolerance by pixel magnitude
        :param alpha: The safety factor, for which :math:`\alpha \leq 1`, reduces chance of rejecting next step size.
        :param h_start: The starting step size.
        :param max_increase: A factor determining the maximum increase of the step size for each step.
        :param max_decrease: A factor determining the maximum decrease of the step size for each step.
        :param interval: The interval over which the SDE is computed
        """
        super().__init__(sde)

        self._ki = ki
        self._kp = kp
        self._tau_a = tau_a
        self._tau_r = tau_r
        self._alpha = alpha
        self._start_time = float(interval[0])
        self._end_time = float(interval[1])
        self._h_start = abs(h_start) * (self._end_time - self._start_time) / abs(self._end_time - self._start_time)
        self._max_increase = torch.Tensor([max_increase])
        self._max_decrease = torch.Tensor([max_decrease])
        self._max_iter = max_iter

    def solve(self, x: torch.Tensor, labels: torch.Tensor = None,
              callback: Callable[[torch.Tensor, torch.Tensor], None] = None) -> torch.Tensor:
        t = broadcast_vector(torch.full((x.shape[0],), self._start_time), x).to(self._device)  # Initialise batch_size times, starting at 1
        h = broadcast_vector(torch.full((x.shape[0],), self._h_start), x).to(self._device)
        error = torch.full((x.shape[0],), 0.5).to(self._device)
        end_condition = broadcast_vector(torch.full((x.shape[0],), self._end_time), x).to(self._device)
        if labels is None:
            labels = torch.zeros(x.shape[0],)
        reject_count = 0
        not_reject_count = 0
        x_full, t_full, h_full = x, t, h
        i = 0

        # Loop until all times in the batch are equal to the end time
        while torch.any(not_finished := (t_full != end_condition)) and i < self._max_iter:
            # Work with the unfinished subset of x, t, h
            not_finished = not_finished.reshape(-1)
            x, t, h = x_full[not_finished], t_full[not_finished], h_full[not_finished]

            # Perform Euler and Heun step
            w = torch.randn_like(x)
            dx_euler = self.sde.step(x, t, h, w, labels=labels[not_finished])
            dx_heun = self.sde.step(x + dx_euler * ((t + h) > 0), t + h * ((t + h) > 0), h, w, labels=labels[not_finished])

            # Compute first and second order x
            x_first = x + dx_euler
            x_second = x + 0.5 * (dx_euler + dx_heun)

            # Compute extrapolated error
            old_error = error.clone()
            new_error = self._error(x_first, x_second)
            error[not_finished] = new_error

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

            if callback is not None:
                callback(x_full, t_full, h_full, error)
            i += 1

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
        if d != 1:
            norm = broadcast_vector(self._tau_a + self._tau_r * torch.std(x_second.view(x_second.shape[0], -1), dim=1), x_second)
        else:
            norm = broadcast_vector(self._tau_a + self._tau_r * torch.abs(x_second), x_second)
        squared_error = torch.square((x_second - x_first) / norm)
        return torch.sqrt(torch.sum(squared_error.view(x_first.shape[0], -1), dim=1) / d)

    def _get_next_step(self, error: torch.Tensor, error_previous, h: torch.Tensor) -> torch.Tensor:
        """
        Computes the next step h for each x based on the error using a PI controller.

        :param error: Current error, tensor of shape (batch_size, 1).
        :param error_previous: Error of previous step, tensor of shape (batch_size, 1).
        :param h: Current step size, tensor of shape (batch_size, 1).
        :return: The new step size, a tensor of shape (batch_size, 1).
        """
        integral = (self._alpha * self._tau_a / error)**(self._ki + self._kp)
        proportional = (error_previous / (self._alpha * self._tau_a))**self._kp
        return h * broadcast_vector(torch.clamp(integral * proportional, min=self._max_decrease, max=self._max_increase), h)

    def to(self, device: str) -> Solver:
        super().to(device)
        self._max_increase = self._max_increase.to(self._device)
        self._max_decrease = self._max_decrease.to(self._device)
        return self


def edm_sampler(
    net, noise, labels=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like,
):
    # Guided denoiser.
    def denoise(x, t):
        Dx = net(x, t, labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t, labels).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = noise.to(dtype) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        d_cur = (x_hat - denoise(x_hat, t_hat)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            d_prime = (x_next - denoise(x_next, t_next)) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next
