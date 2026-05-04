"""Various Variance Exploding SDEs"""
from typing import Callable
import math

import torch

from sde_lib.sde import LinearDriftSDE


class VarianceExplodingSDE(LinearDriftSDE):
    r"""
    Implements the Variance Exploding SDE by Song et al. (2021). This SDE is given by
    :math:`dx = \sqrt{\frac{d[\sigma(t)^2]}{dt}dw}`
    With :math:`\sigma(t) = \sigma_{min}(\frac{\sigma_{max}}{\sigma_{min}})^t`
    """

    def __init__(self, sigma_min: float, sigma_max: float):
        """
        Constructs the VarianceExplodingSDE.

        :param sigma_min: The standard deviation, at t = 0
        :param sigma_max: The standard deviation at t = 1
        """
        super().__init__()

        self._sigma_min = sigma_min
        self._sigma_max = sigma_max

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self._sigma_min * (self._sigma_max / self._sigma_min) ** t

    def mu(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.ones(1).to(self._device)

    def drift(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        return torch.zeros(x.shape).to(self._device)

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        # Compute derivative of sigma^2
        # ds^2(t)/dt = 2 * s^2(t) * (ln(s_max) - ln(s_min))
        dsigma_dt = 2 * torch.square(self.sigma(t)) * (math.log(self._sigma_max) - math.log(self._sigma_min))
        return torch.sqrt(dsigma_dt)


class EDMSDE(LinearDriftSDE):

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(2 * t)

    def drift(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        return torch.zeros(x.shape).to(self._device).to(self._device)

    def mu(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape).to(self._device)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def get_reverse_sde(self, denoiser: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor], ode_threshold: float = 0) -> 'SDE':
        def score_fn(x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
            d = denoiser(x, t, labels)
            return (d - x) / torch.square(t)

        return super().get_reverse_sde(score_fn, ode_threshold)


class VarianceExplodingEDMSDE(VarianceExplodingSDE):

    def get_reverse_sde(self, denoiser: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor], ode_threshold: float = 0) -> 'SDE':
        def score_fn(x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
            sigma = self.sigma(t)
            return (denoiser(x, sigma, labels) - x) / torch.square(sigma)

        return super().get_reverse_sde(score_fn, ode_threshold)