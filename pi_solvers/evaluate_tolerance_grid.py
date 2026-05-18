import pathlib
import re
import random
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import tqdm

from pi_solvers.generate_tolerance_nfe_relation import create_grid
from pi_solvers.utils import plot_images, Metric
from pi_solvers.evaluation import feature_vector


class Rater(ABC):

    @abstractmethod
    def rate(self, eval_point, images: list[str]) -> float:
        pass


class ManualRater(Rater):

    def __init__(self, n_cols: int):
        self.n_cols = n_cols

    def rate(self, eval_point, images: list[str]) -> float:
        # Plot images
        plot_images(images, self.n_cols).show()

        # Get image rating
        while True:
            try:
                rating = int(input("What is the quality of the following images (1-5):\n"))
                assert 0 < rating <= 5
                break
            except (ValueError, AssertionError):
                print("Please provide an integer from 1 to 5")

        return float(rating)


class MetricRater(Rater):

    def __init__(
            self,
            metric: Metric,
            ref: torch.Tensor,
            batch_size: int = 64,
            device: str | torch.device = "cuda"
    ):
        self._metric = metric
        self._ref = ref
        self._batch_size = batch_size
        self._device = device

    def rate(self, eval_point: str, images: list[str]) -> float:
        features = feature_vector.detect_image_features(
            image_dir=eval_point, batch_size=self._batch_size, device=self._device
        )
        return self._metric.get_func()(self._ref, features)


def image_path_iterator(target_path: pathlib.Path) -> list[tuple[str, list[str]]]:
    subdirs = []
    for subdir in target_path.iterdir():
        if not subdir.is_file():
            images = []
            for image_path in pathlib.Path(subdir).iterdir():
                images.append(image_path)
            subdirs.append((str(subdir), images))
    return subdirs


def evaluate_images(target_path: str, rater: Rater, seed = 42):
    random.seed(seed)

    df = pd.DataFrame(columns=["tau_a", "tau_r", "rating"])

    paths = image_path_iterator(pathlib.Path(target_path))
    shuffled_paths = random.sample(paths, len(paths))

    for i, (eval_point, images) in tqdm.tqdm(enumerate(shuffled_paths)):
        rating = rater.rate(eval_point, images)

        # Extract tau_a, tau_r from eval point
        tau_a = float(re.search(r"abs_\d+.?\d*", eval_point).group(0)[4:])
        tau_r = float(re.search(r"rel_\d+.?\d*", eval_point).group(0)[4:])

        # Write to df in form tau_a, tau_r, rating
        df.loc[i] = [tau_a, tau_r, rating]

    df.to_csv(target_path + "/ratings.csv")
    return df


def plot_ratings(tau_a_range: tuple[float, float], tau_r_range: tuple[float, float], resolution: int, df: pd.DataFrame, outdir: str):
    av, rv = create_grid(tau_a_range, tau_r_range, resolution)
    ratings = torch.zeros_like(av)

    for i in range(av.shape[1]):
        for j in range(av.shape[0]):
            tau_a, tau_r = round(float(av[i, j]), 3), round(float(rv[i, j]), 3)
            ratings[i, j] = np.log(df[np.logical_and(df["tau_a"] == tau_a, df["tau_r"] == tau_r)]["rating"].iloc[0])

    plt.figure()
    mesh = plt.pcolormesh(av, rv, ratings, cmap='inferno')
    plt.colorbar(mesh, label='Log MIND')
    plt.xlabel(r"$\tau_a$")
    plt.ylabel(r"$\tau_r$")
    plt.title(r"Ratings as a function of $\tau_a$ and $\tau_r$")
    plt.savefig(outdir + "/ratings.png")


if __name__ == "__main__":
    rater = MetricRater(Metric.MIND, ref=torch.load("../refs/img64_features.pkl"))

    target = "../data/tolerance_grid/eval_pisolver_2"
    # df = evaluate_images(target, rater)
    df = pd.read_csv(target + "/ratings.csv")
    plot_ratings((0.05, 0.5), (0.5, 10), 20, df, target)