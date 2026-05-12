from typing import Callable
import pickle

import matplotlib.pyplot as plt
import torch
from torchvision.models import inception_v3, Inception_V3_Weights

import dnnlib


def broadcast_vector(vector: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    return vector.view(tensor.shape[0], *([1] * (tensor.dim() - 1)))


def load_edm_checkpoint(url: str) -> tuple[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    # Load model
    with dnnlib.util.open_url(url) as f:
        data = pickle.load(f)
    model = data["ema"]

    # Load encoder
    encoder = data.get('encoder', None)
    if encoder is None:
        encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')

    return model, encoder


def calculate_fid(X: torch.Tensor, X_hat: torch.Tensor) -> float:
    mu, sigma         = X.mean(dim=0), X.cov()
    mu_hat, sigma_hat = X_hat.mean(dim=0), X_hat.cov()
    mu_diff = torch.sum((mu - mu_hat)**2)
    covs = sigma + sigma_hat - 2 * ((sigma * sigma_hat)**0.5).real
    return float(mu_diff + torch.trace(covs))


def get_feature_vectors(X: torch.Tensor) -> float:
    return inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, progress=True)(X)


def plot_images(images: list[str], n_cols: int) -> plt.Figure:
    # Plot images
    fig, axes = plt.subplots(len(images) // n_cols, n_cols)
    fig.set_size_inches(10, 10)

    for index, image_path in enumerate(images):
        j = index % n_cols
        k = index // n_cols

        image_arr = plt.imread(image_path)

        axes[j][k].imshow(image_arr)
        axes[j][k].axis("off")

    fig.tight_layout()
    return fig
