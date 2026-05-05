import torch
from torchvision.models import inception_v3, Inception_V3_Weights
import matplotlib.pyplot as plt


def broadcast_vector(vector: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    return vector.view(tensor.shape[0], *([1] * (tensor.dim() - 1)))


def calculate_fid(X: torch.Tensor, X_hat: torch.Tensor) -> float:
    mu, sigma         = X.mean(dim=0), X.cov()
    mu_hat, sigma_hat = X_hat.mean(dim=0), X_hat.cov()
    mu_diff = torch.sum((mu - mu_hat)**2)
    covs = sigma + sigma_hat - 2 * ((sigma * sigma_hat)**0.5).real
    return float(mu_diff + torch.trace(covs))


def get_feature_vectors(X: torch.Tensor) -> float:
    return inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, progress=True)(X)
