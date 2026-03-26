import torch
import math
from typing import Callable
from sde import LinearDriftSDE


class Gaussian:

    def __init__(self, mu: torch.Tensor | float, sigma: torch.Tensor | float, norm: torch.Tensor | float = None , weight: torch.Tensor | float = None):
        self._mu = mu
        self._sigma = sigma
        self._norm = norm
        if not self._norm:
            self._norm = 1 / (sigma * math.sqrt(2 * math.pi))

            if weight:
                self._norm *= weight


    @property
    def mu(self):
        return self._mu

    @property
    def sigma(self):
        return self._sigma

    @property
    def norm(self):
        return self._norm

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return - (x - self.mu) / self.sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm * torch.exp(-0.5 * torch.square(x - self.mu) / self.sigma)


class MultiGaussian:

    def __init__(self, gaussians: tuple[Gaussian, ...], sde: LinearDriftSDE):
        self._gaussians = gaussians
        self._sde = sde

    def get_score_function(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        return self.score

    def score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        convolved_gaussians = self.convolve_gaussians(x, t)
        non_normalised_score = sum(map(lambda gaussian: gaussian.score(x) * gaussian(x), convolved_gaussians))
        normalisation = sum(map(lambda gaussian: gaussian(x), convolved_gaussians))
        return torch.Tensor(non_normalised_score / normalisation)

    def convolve_gaussians(self, x: torch.Tensor, t: torch.Tensor) -> tuple[Gaussian, ...]:
        convolved_gaussians = []
        alpha_t, sigma_t = self._sde.marginal(x, t)

        for gaussian in self._gaussians:
            mu_new = gaussian.mu * alpha_t
            sigma_new = torch.sqrt(alpha_t**2 * gaussian.sigma**2 + sigma_t**2)
            convolved_gaussians.append(Gaussian(mu_new, sigma_new, gaussian.norm))

        return tuple(convolved_gaussians)
