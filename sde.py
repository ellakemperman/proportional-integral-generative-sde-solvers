"""Definitions of the stochastic differential equations"""
from abc import abstractmethod, ABC
import torch
from typing import Callable


class SDE(ABC):

    def __init__(self):
        pass

    def sde(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the Ito parameters of the SDE for a given sample and time. The SDE is defined by the drift f(t) and
        the diffusion g(t).

        :param x: The given sample.
        :param t: The given time.
        :return: A tuple of (drift, diffusion)
        """
        return self._drift(x, t), self._diffusion(t)

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the Ito parameters of the SDE for a given sample and time. The SDE is defined by the drift f(t) and
        the diffusion g(t).

        :param x: The given sample.
        :param t: The given time.
        :return: A tuple of (drift, diffusion)
        """
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


class LinearDriftSDE(SDE, ABC):

    def marginal(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the Gaussian marginal probability distribution of a given sample, at a given time t.
        Marginal distribution can be obtained by solving the Focker-Planck Equation.

        :param x: The given sample.
        :param t: The time
        :return: A tuple of (mu, sigma), representing the mean and standard deviation of the sample.
        """
        return self._mu(t), self._sigma(t)

    def sample_noise(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, sigma = self.marginal(x, t)
        noise = torch.randn_like(x)
        return mu * x + sigma * torch.randn_like(x), noise

    def sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.sample_noise(x, t)[0]

    @abstractmethod
    def _mu(self, t: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        pass


class VarianceExplodingSDE(LinearDriftSDE):

    def __init__(self, sigma_min: float, sigma_max: float):
        super().__init__()
        self._sigma_min = torch.Tensor(sigma_min)
        self._sigma_max = torch.Tensor(sigma_max)

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.square(self._sigma_min * (self._sigma_max / self._sigma_min) ** t)

    def _mu(self, t: torch.Tensor) -> torch.Tensor:
        return torch.Tensor(1)

    def _drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape)

    def _diffusion(self, t: torch.Tensor) -> torch.Tensor:
        # Compute derivative of sigma^2
        # ds^2(t)/dt = 2 * s^2(t) * (ln(s_max) - ln(s_min))
        dsigma_dt = 2 * torch.sqrt(self._sigma(t)) * (torch.log(self._sigma_max) - torch.log(self._sigma_min))
        return torch.sqrt(dsigma_dt)
