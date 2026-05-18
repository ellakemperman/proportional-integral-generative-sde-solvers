import enum
import pathlib
from typing import Callable
import pickle

import matplotlib.pyplot as plt
import torch
from torchvision.io import decode_image

from torchvision.models import inception_v3, Inception_V3_Weights

from pi_solvers import dnnlib
from pi_solvers.torch_utils.dataset import Dataset
from pi_solvers.evaluation import metrics


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


class ImageSampleDataset(Dataset):

    def __init__(self, image_dir: str, n_images: int = 0, transform = None):
        self._images = list(pathlib.Path(image_dir).iterdir())
        if n_images:
            self._images = self._images[:n_images]

        self._n_images = n_images
        self._transform = transform

    def __len__(self):
        return len(self._images)

    def __getitem__(self, item):
        image = decode_image(self._images[item])
        if self._transform:
            image = self._transform(image)
        return image


###
# From https://github.com/NVlabs/edm2/blob/main/generate_images.py
class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])
###


class Metric(enum.Enum):
    FID = "FID"
    MIND = "MIND"

    def __str__(self):
        return self.value

    def get_func(self):
        match self.value:
            case "FID": return metrics.frechet_inception_distance
            case "MIND": return metrics.monge_inception_distance

    def uses_stats(self):
        if self == Metric.FID:
            return True
        return False
