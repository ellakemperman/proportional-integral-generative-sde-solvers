import os
import pathlib
from typing import Callable
import pickle

import matplotlib.pyplot as plt
import torch
from torch.utils.data import IterableDataset
from torchvision.io import decode_image

from torchvision.models import inception_v3, Inception_V3_Weights

from pi_solvers import dnnlib


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


class ImageSampleDataset(IterableDataset):

    def __init__(self, image_dir: str, n_images: int = 0, transform = None):
        super().__init__()
        self._image_dir = image_dir
        self._n_images = n_images
        self._transform = transform

    def __len__(self):
        return len(os.listdir(self._image_dir)) if not self._n_images else self._n_images

    def __iter__(self):
        for i, image_path in enumerate(pathlib.Path(self._image_dir).iterdir()):
            if i >= self._n_images and self._n_images:
                break

            image = decode_image(image_path)
            if self._transform:
                image = self._transform(image)
            yield image
