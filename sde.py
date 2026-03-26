"""Definitions of the stochastic differential equations"""
from abc import abstractmethod, ABC
from typing import Callable
import torch


class SDE(ABC):
    r"""
    The Stochastic Differential Equation (SDE) abstract class. Assumes the SDE can be written in Ito form:
    :math:`dx = f(x, t)dt + g(t)dw`
    Where :math:`f(x,t)` is the drift, and :math:`g(t)` the diffusion.
    """

    def sde(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the Ito parameters of the SDE for a given sample and time. The SDE is defined by the drift f(t) and
        the diffusion g(t).

        :param x: The given sample.
        :param t: The given time.
        :return: A tuple of (drift, diffusion)
        """
        return self.drift(x, t), self.diffusion(t)

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the Ito parameters of the SDE for a given sample and time. The SDE is defined by the drift f(t) and
        the diffusion g(t).

        :param x: The given sample.
        :param t: The given time.
        :return: A tuple of (drift, diffusion)
        """
        return self.sde(x, t)

    def step(self, x: torch.tensor, t: torch.tensor, dt: torch.tensor, w: torch.Tensor = None) -> torch.Tensor:
        r"""
        Takes a step of size dt.

        :param x: The x value at time t
        :param t: The time
        :param dt: The step size
        :param w: The noise added. By default, this is set to :math:`\mathcal{N}(0, I)`. Provide if specific noise
                  needs to be added.
        :return: x + dx
        """
        if not w:
            w = torch.randn_like(x)
        drift, diffusion = self.sde(x, t)
        return x + drift * dt + diffusion * torch.sqrt(torch.abs(dt)) * w

    @abstractmethod
    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Gives the drift f(x, t) at given x, t of the SDE.

        :param x: The given sample
        :param t: The given time
        :return: f(x, t)
        """
        pass

    @abstractmethod
    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """
        Gives the diffusion g(t) at a given time t.

        :param t: The time
        :return: g(t)
        """
        pass

    def get_reverse_sde(self, score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> 'SDE':
        r"""
        Reverses the SDE to the form :math:`dx = (f(x, t) - g(t)^2 \nabla_x \log p_t(x))dt + g(t)dw`

        :param score_fn: The score function, a function s(x, t) for which :math:`s(x, t) \approx \nabla_x \log p_t(x)`
        :return: The reversed SDE
        """
        parent = self

        # Construct ReverseSDE class as child from caller SDE, use parent drift and diffusion but update drift to
        # include the score
        class ReverseSDE(parent.__class__):
            """A reversed SDE."""

            def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                return parent.drift(x, t) - torch.square(parent.diffusion(t)) * score_fn(x, t)

            def diffusion(self, t: torch.Tensor) -> torch.Tensor:
                return parent.diffusion(t)

        return ReverseSDE()


class LinearDriftSDE(SDE, ABC):
    r"""
    This SDE represents a class of SDEs with drift linear in x. Specifically, :math:`f(x, t) = F(t)x` holds
    for this class of SDEs. This type of SDE has an analytical solution for the marginal.
    """

    def marginal(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the Gaussian marginal probability distribution of a given sample, at a given time t.
        Marginal distribution can be obtained by solving the Focker-Planck Equation.

        :param x: The given sample.
        :param t: The time
        :return: A tuple of (mu, sigma), representing the mean and standard deviation of the sample.
        """
        return self.mu(x, t), self.sigma(t)

    def sample_noise(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes a sample x from the marginal :math:`p(x_t|x_0)`, returning the noise for training convenience.

        :param x: The sample x.
        :param t: Time t from which should be sampled.
        :return: A tuple of (:math:`x \sim p(x_t|x_0), \epsilon`)
        """
        mu, sigma = self.marginal(x, t)
        noise = torch.randn_like(x)
        return mu * x + sigma * torch.randn_like(x), noise

    def sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""
        Computes a sample x from the marginal :math:`p(x_t|x_0)`.

        :param x: The sample x.
        :param t: Time t from which should be sampled.
        :return: :math:`x \sim p(x_t|x_0)`
        """
        return self.sample_noise(x, t)[0]

    @abstractmethod
    def mu(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        pass


class VarianceExplodingSDE(LinearDriftSDE):
    r"""
    Implements the Variance Exploding SDE by Song et al. (2021). This SDE is given by
    :math:`dx = \sqrt{\frac{d[\sigma(t)^2]}{dt}dw}`
    With :math:`\sigma(t) = \sigma_{min}(\frac{\sigma_{max}}{\sigma_{min}})^t`
    """

    def __init__(self, sigma_min: float, sigma_max: float):
        self._sigma_min = torch.Tensor(sigma_min)
        self._sigma_max = torch.Tensor(sigma_max)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.square(self._sigma_min * (self._sigma_max / self._sigma_min) ** t)

    def mu(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.ones(1)

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape)

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        # Compute derivative of sigma^2
        # ds^2(t)/dt = 2 * s^2(t) * (ln(s_max) - ln(s_min))
        dsigma_dt = 2 * torch.sqrt(self.sigma(t)) * (torch.log(self._sigma_max) - torch.log(self._sigma_min))
        return torch.sqrt(dsigma_dt)


class VariancePreservingSDE(LinearDriftSDE, ABC):

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - torch.exp(-self._B(t))

    def mu(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * self._B(t))

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -0.5 * self._beta(t) * x

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self._beta(t))

    @abstractmethod
    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _B(self, t: torch.Tensor) -> torch.Tensor:
        pass


class LinearVariancePreservingSDE(VariancePreservingSDE):

    def __init__(self, beta_min: float, beta_max: float):
        self._beta_min = beta_min
        self._beta_max = beta_max

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        return self._beta_min + t * (self._beta_max - self._beta_min)

    def _B(self, t: torch.Tensor) -> torch.Tensor:
        return self._beta_min * t + 0.5 * torch.square(t) * (self._beta_max - self._beta_min)
