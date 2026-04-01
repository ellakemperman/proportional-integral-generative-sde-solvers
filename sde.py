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

    def __init__(self):
        self._nfe = 0

    @property
    def nfe(self):
        return self._nfe

    def reset_nfe(self):
        self._nfe = 0

    def sde(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the Ito parameters of the SDE for a given sample and time. The SDE is defined by the drift f(t) and
        the diffusion g(t).

        :param x: The given sample, a tensor of shape (batch_size, d)
        :param t: The given time, a tensor of shape (batch_size, 1)
        :return: A tuple of (drift, diffusion)
        """
        self._nfe += x.shape[0]  # Add batch_size to NFE
        return self.drift(x, t), self.diffusion(t)

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the Ito parameters of the SDE for a given sample and time. The SDE is defined by the drift f(t) and
        the diffusion g(t).

        :param x: The given sample, a tensor of shape (batch_size, d).
        :param t: The given time, a tensor of shape (batch_size, 1).
        :return: A tuple of (drift, diffusion), with shapes (batch_size, d) and (batch_size, 1).
        """
        return self.sde(x, t)

    def step(self, x: torch.tensor, t: torch.tensor, dt: torch.tensor, w: torch.Tensor = None) -> torch.Tensor:
        r"""
        Takes a step of size dt.

        :param x: The given sample, a tensor of shape (batch_size, d)
        :param t: The given time, a tensor of shape (batch_size, 1)
        :param dt: The step size, a tensor of shape (batch_size, 1)
        :param w: The noise added. By default, this is set to :math:`\mathcal{N}(0, I)`. Provide if specific noise
                  needs to be added. A tensor of shape (batch_size, d)
        :return: x + dx, a tensor of shape (batch_size, d)
        """
        if w is None:
            w = torch.randn_like(x)
        drift, diffusion = self.sde(x, t)
        return drift * dt + diffusion * torch.sqrt(torch.abs(dt)) * w

    @abstractmethod
    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Gives the drift f(x, t) at given x, t of the SDE.

        :param x: The given sample, a tensor of shape (batch_size, d)
        :param t: The given time, a tensor of shape (batch_size, 1)
        :return: f(x, t), a tensor of shape (batch_size, d)
        """
        pass

    @abstractmethod
    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """
        Gives the diffusion g(t) at a given time t.

        :param t: The given time, a tensor of shape (batch_size, 1)
        :return: g(t), a tensor of shape (batch_size, 1)
        """
        pass

    def get_reverse_sde(self, score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> 'SDE':
        r"""
        Reverses the SDE to the form :math:`dx = (f(x, t) - g(t)^2 \nabla_x \log p_t(x))dt + g(t)dw`

        :param score_fn: The score function, a function s(x, t) for which :math:`s(x, t) \approx \nabla_x \log p_t(x)`
                         which takes a tensor of shape (batch_size, d) and (batch_size, 1) and maps it to another
                         tensor of (batch_size, d)
        :return: The reversed SDE
        """
        parent = self

        # Construct ReverseSDE class as child from SDE, use parent drift and diffusion but update drift to
        # include the score
        class ReverseSDE(SDE):
            """A reversed SDE."""
            def __init__(self):
                super().__init__()

                self._parent = parent

            @property
            def parent(self) -> 'SDE':
                return parent

            def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                return self.parent.drift(x, t) - torch.square(self.parent.diffusion(t)) * score_fn(x, t)

            def diffusion(self, t: torch.Tensor) -> torch.Tensor:
                return self.parent.diffusion(t)

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

        :param x: The given sample, a tensor of shape (batch_size, d)
        :param t: The given time, a tensor of shape (batch_size, 1)
        :return: A tuple of (mu, sigma), representing the mean and standard deviation of the sample. mu is a tensor
                 of shape (batch_size, d), sigma a tensor of shape (batch_size, 1)
        """
        return self.mu(x, t), self.sigma(t)

    def sample_noise(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes a sample x from the marginal :math:`p(x_t|x_0)`, returning the noise for training convenience.

        :param x: The given sample, a tensor of shape (batch_size, d)
        :param t: The times t from which should be sampled, a tensor of shape (batch_size, 1)
        :return: A tuple of (:math:`x \sim p(x_t|x_0), \epsilon`), both with shape (batch_size, d)
        """
        mu, sigma = self.marginal(x, t)
        noise = torch.randn_like(x)
        return mu * x + sigma * torch.randn_like(x), noise

    def sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""
        Computes a sample x from the marginal :math:`p(x_t|x_0)`.

        :param x: The given sample, a tensor of shape (batch_size, d)
        :param t: The times t from which should be sampled, a tensor of shape (batch_size, 1)
        :return: :math:`x \sim p(x_t|x_0)`, with shape (batch_size, d)
        """
        return self.sample_noise(x, t)[0]

    @abstractmethod
    def mu(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the marginal mean of x at time t.

        :param x: The given sample, a tensor of shape (batch_size, d)
        :param t: The times t from which should be sampled, a tensor of shape (batch_size, 1)
        :return: The marginal mean of x at time t, a tensor of shape (batch_size, d)
        """
        pass

    @abstractmethod
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the marginal standard deviation at time t.

        :param t: The times t from which should be sampled, a tensor of shape (batch_size, 1)
        :return: The marginal standard deviation at time t, a tensor of shape (batch_size, d)
        """
        pass


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
    r"""
    Implements the Variance Preserving SDE by Song et al. (2021). This SDE is given by
    :math:`dx = -\frac{1}{2}\beta(t)xdt + \sqrt{\beta(t)}dw`, where :math:`\beta(t)` is any differentiable function of t.
    """

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
        r"""
        The beta function, a differentiable function of t.

        :param t: The times t, a tensor of shape (batch_size, 1)
        :return: :math:`\beta(t)`
        """
        pass

    @abstractmethod
    def _B(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        The antiderivative of the beta function, :math:`B(t) = \int \beta(t)dt`

        :param t: The times t, a tensor of shape (batch_size, 1)
        :return: :math:`B(t)`
        """
        pass


class LinearVariancePreservingSDE(VariancePreservingSDE):
    r"""
    A Variance Preserving SDE with a linear beta function, given by
    :math:`\beta(t) = (\beta_{max} - \beta_{min}) * t + \beta_{min}`
    """

    def __init__(self, beta_min: float, beta_max: float):
        r"""
        Constructs the LinearVariancePreservingSDE.

        :param beta_min: Minimum value of :math:`\beta`.
        :param beta_max: Maximum value of :math:`\beta`.
        """
        super().__init__()

        self._beta_min = beta_min
        self._beta_max = beta_max

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        return self._beta_min + t * (self._beta_max - self._beta_min)

    def _B(self, t: torch.Tensor) -> torch.Tensor:
        return self._beta_min * t + 0.5 * torch.square(t) * (self._beta_max - self._beta_min)
