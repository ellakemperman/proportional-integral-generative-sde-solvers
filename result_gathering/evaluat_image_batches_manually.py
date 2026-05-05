import pathlib
import re
import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch


from generate_tolerance_nfe_relation import create_grid


def image_path_iterator(target_path: pathlib.Path) -> list[tuple[str, list[str]]]:
    subdirs = []
    for subdir in target_path.iterdir():
        if not subdir.is_file():
            images = []
            for image_path in pathlib.Path(subdir).iterdir():
                images.append(image_path)
            subdirs.append((str(subdir), images))
    return subdirs


def evaluate_images(target_path: str, n_cols: int = 4, seed = 42):
    random.seed(seed)

    df = pd.DataFrame(columns=["tau_a", "tau_r", "rating"])

    paths = image_path_iterator(pathlib.Path(target_path))
    shuffled_paths = random.sample(paths, len(paths))

    for i, (eval_point, images) in enumerate(shuffled_paths):
        # Plot images
        fig, axes = plt.subplots(len(images) // n_cols, n_cols)
        for index, image_path in enumerate(images):
            j = index % n_cols
            k = index // n_cols

            image_arr = plt.imread(image_path)

            axes[j][k].imshow(image_arr)
            axes[j][k].axis("off")

        fig.tight_layout()
        fig.show()

        # Get image rating
        while True:
            try:
                rating = int(input("What is the quality of the following images (1-5):\n"))
                assert 0 < rating <= 5
                break
            except (ValueError, AssertionError):
                print("Please provide an integer from 1 to 5")

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
            ratings[i, j] = df[np.logical_and(df["tau_a"] == tau_a, df["tau_r"] == tau_r)]["rating"].iloc[0]

    plt.figure()
    mesh = plt.pcolormesh(av, rv, ratings, cmap='inferno')
    plt.colorbar(mesh, label='Ratings')
    plt.xlabel(r"$\tau_a$")
    plt.ylabel(r"$\tau_r$")
    plt.title(r"Ratings as a function of $\tau_a$ and $\tau_r$")
    plt.savefig(outdir + "/ratings.png")


if __name__ == "__main__":
    target = "../data/tolerance_grid/eval"
    # evaluate_images(target, 4)
    df = pd.read_csv(target + "/ratings.csv")
    plot_ratings((0.1, 1), (0.5, 10), 20, df, target)