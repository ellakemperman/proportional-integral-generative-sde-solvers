"""Computes metrics of the data"""
import enum

import torch
from scipy.linalg import sqrtm
import numpy as np


def monge_inception_distance(
        x: torch.Tensor,
        x_hat: torch.Tensor,
        seed: int = 0,
        n_projections: int = 100
):
    """
    Computes the MIND metric from the paper MIND: Monge Inception Distance for Generative Models
    Evaluation: https://arxiv.org/html/2605.06797v1#A3.F10.sf2

    :param x: Ground truth
    :param x_hat: The generated features
    :param seed: Random seed
    :param n_projections: Number of projections to use
    :return: The value of the MIND metric
    """
    generator = torch.Generator(device=x.device).manual_seed(seed)

    n_samples, d = x_hat.shape
    assert n_samples <= x.shape[0], "Ground truth needs to have at least as many samples as predicted"

    x = x[torch.randint(x.shape[0], size=(n_samples,), generator=generator)]

    alpha = 3 * d

    u_proj = torch.randn(
        (n_projections, d),
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
        )**2,
        dim=-1
    )

    return float(alpha * torch.mean(dists))


def frechet_inception_distance(
        x: torch.Tensor | dict,
        x_hat: torch.Tensor,
        dtype=torch.float64
) -> float:
    """
    Computes the frechet inception distance.

    :param x: Feature vectors of the reference dataset.
    :param x_hat: Feature vectors of the generated samples.
    :param dtype: Torch dtype
    :return: The value of the Frechet Inception Distance
    """
    x_hat = x_hat.to(dtype)

    if isinstance(x, dict):
        mu, sigma = x["mu"], x["sigma"]
    else:
        x = x.to(dtype)
        mu, sigma = calc_mean_covariance(x)

    mu_hat, sigma_hat = calc_mean_covariance(x_hat)

    mu_diff = np.sum((mu - mu_hat)**2)
    s, _ = sqrtm(sigma @ sigma_hat, disp=False)
    covs = sigma + sigma_hat - 2 * s
    return float(np.real(mu_diff + np.trace(covs)))


def calc_mean_covariance(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    return x.mean(dim=0).cpu().numpy(), x.T.cov().cpu().numpy()


class Metric(enum.Enum):
    FID = "FID"
    MIND = "MIND"

    def __str__(self):
        return self.value

    def get_func(self):
        match self.value:
            case "FID": return frechet_inception_distance
            case "MIND": return monge_inception_distance

    def uses_stats(self):
        if self == Metric.FID:
            return True
        return False
