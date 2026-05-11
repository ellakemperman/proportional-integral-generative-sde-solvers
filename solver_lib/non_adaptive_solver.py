"""Definitions of non-adaptive solvers, includes Euler-Marayuma, Heun, and EDM solver."""
from typing import Callable
import torch

from sde_lib import SDE
from utils import broadcast_vector

from solver_lib.solvers import Solver


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
            w = torch.randn_like(x)
            dx_euler = self.sde.step(x, t, dt, w=w, labels=labels)

            if i < (self._time_steps.shape[0] - 1):
                dx_heun = self.sde.step(x + dx_euler, t + dt, dt, w=w, labels=labels)
                x += 0.5 * (dx_euler + dx_heun)
            else:
                x += dx_euler

            if callback is not None:
                callback(x, t + dt)

        return x


def get_edm_schedule(
        n_steps: int,
        t_min: float = 0,
        t_max: float = 80,
        rho: float = 7
):
    step_indices = torch.arange(n_steps)
    t_steps = (t_max ** (1 / rho) + step_indices / (n_steps - 1)
               * (t_min ** (1 / rho) - t_max ** (1 / rho))) ** rho
    discretisation = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    return discretisation


# From EDM2: https://github.com/NVlabs/edm2/tree/main
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
    t_steps = get_edm_schedule(num_steps, sigma_min, sigma_max, rho).to(noise.device)

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