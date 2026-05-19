"""Computes metrics of the data"""
import enum
from abc import abstractmethod, ABC

import torch
from scipy.linalg import sqrtm
import numpy as np


class Metrics(enum):
    MIND = MIND()
    FID = FID()

    def uses_stats(self):
        return self.value.uses_stats()

    def pretty_print(self, x: torch.Tensor, x_hat: torch.Tensor):
        return self.value.pretty_print(x, x_hat)

    def __call__(self, x: torch.Tensor, x_hat: torch.Tensor):
        return self.value(x, x_hat)

    def __str__(self):
        return str(self.value)


class Metric(ABC):

    def pretty_print(self, x: torch.Tensor, x_hat: torch.Tensor) -> str:
        return str(self) + ": " + str(round(self(x, x_hat), 4))

    def uses_stats(self) -> bool:
        return False

    @abstractmethod
    def __call__(self, x: torch.Tensor, x_hat: torch.Tensor):
        pass


class MIND(Metric):
    """
    Computes the MIND metric from the paper MIND: Monge Inception Distance for Generative Models
    Evaluation: https://arxiv.org/html/2605.06797v1#A3.F10.sf2
    """

    def __init__(self, seed: int = 0, n_projections: int = 0):
        self._seed = seed
        self._n_projections = n_projections

    def __call__(self, x, x_hat):
        generator = torch.Generator(device=x.device).manual_seed(self._seed)

        n_samples, d = x_hat.shape
        assert n_samples <= x.shape[0], "Ground truth needs to have at least as many samples as predicted"

        x = x[torch.randint(x.shape[0], size=(n_samples,), generator=generator)]

        alpha = 3 * d

        u_proj = torch.randn(
            (self._n_projections, d),
            generator=generator,
            dtype=x.dtype,
            device=x.device
        )

        u_proj /= torch.linalg.norm(u_proj, dim=-1, keepdim=True)

        x_proj = u_proj @ x.T
        x_hat_proj = u_proj @ x_hat.T

        dists = torch.mean(
            (
                    torch.topk(x_hat_proj, n_samples, dim=-1).values -
                    torch.topk(x_proj, n_samples, dim=-1).values
            ) ** 2,
            dim=-1
        )

        return float(alpha * torch.mean(dists))

    def __str__(self):
        return "MIND"


class FID(Metric):

    def __init__(self, dtype = torch.float64):
        self._dtype = dtype

    def uses_stats(self) -> bool:
        return True

    def __call__(self, x, x_hat):
        """
        Computes the frechet inception distance.

        :param x: Feature vectors of the reference dataset.
        :param x_hat: Feature vectors of the generated samples.
        :param dtype: Torch dtype
        :return: The value of the Frechet Inception Distance
        """
        x_hat = x_hat.to(self._dtype)

        if isinstance(x, dict):
            mu, sigma = x["mu"], x["sigma"]
        else:
            x = x.to(self._dtype)
            mu, sigma = self.calc_mean_covariance(x)

        mu_hat, sigma_hat = self.calc_mean_covariance(x_hat)

        mu_diff = np.sum((mu - mu_hat) ** 2)
        s, _ = sqrtm(sigma @ sigma_hat, disp=False)
        covs = sigma + sigma_hat - 2 * s
        return float(np.real(mu_diff + np.trace(covs)))

    @staticmethod
    def calc_mean_covariance(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        return x.mean(dim=0).cpu().numpy(), x.T.cov().cpu().numpy()
