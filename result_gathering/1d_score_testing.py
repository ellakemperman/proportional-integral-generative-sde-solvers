"""Tests various solvers on a 1D Gaussian."""
from typing import Iterable, Callable, Any
import shutil
import os
import time
import math
import tqdm

import pandas as pd
import torch
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import solver_lib
import sde_lib
from utils import gaussians


def calculate_distance(x1: torch.Tensor, x2: torch.Tensor, n_bins=1000) -> float:
    bin_min, bin_max = torch.min(x1[0], x2[0]), torch.max(x1[-1], x2[-1])
    bin_cutoffs = torch.linspace(bin_min, bin_max, n_bins)
    return float(torch.sum(torch.abs(torch.histogram(x1, bins=bin_cutoffs)[0] - torch.histogram(x2, bins=bin_cutoffs)[0])) / x1.shape[0])


def evaluate_solvers(solvers: Iterable[solver_lib.Solver],
                     x_start: torch.Tensor,
                     x: torch.Tensor,
                     sde_: sde_lib.SDE,
                     seed: int = 0
                     ) -> tuple[list[float], list[float]]:
    nfe, distance = [], []
    for solver in tqdm.tqdm(solvers):
        torch.manual_seed(seed)
        sde_.reset_nfe()
        try:
            x_hat = solver.solve(x_start.clone())
            nfe.append(sde_.nfe / x.shape[0])
            distance.append(
                calculate_distance(x[:, 0], torch.sort(x_hat)[0][:, 0])
            )
        except TimeoutError:
            print("Timeout error")
            nfe.append(-1)
            distance.append(-1)
    return nfe, distance


def create_solvers(
        constructor: Callable[[Any], solver_lib.Solver],
        parameter_range: Iterable[Any]
) -> Iterable[solver_lib.Solver]:
    for param in parameter_range:
        yield constructor(param)


if __name__ == "__main__":
    seed = 0
    write_path = f"../data/gaussian_results/data-{round(time.time(), 0) % 1e10}/"
    torch.manual_seed(seed)

    # Save settings
    os.mkdir(write_path[:-1])
    shutil.copy(__file__, write_path + os.path.basename(__file__))

    # SDE Setup
    beta_min = 0.1
    beta_max = 20
    sde = sde_lib.LinearVariancePreservingSDE(beta_min, beta_max)

    # Gaussian & score setup
    gaussian1 = gaussians.Gaussian(
        mu=0,
        sigma=1,
        weight=1 / 8
    )
    gaussian2 = gaussians.Gaussian(
        mu=-50,
        sigma=3,
        weight=2 / 8
    )

    gaussian3 = gaussians.Gaussian(
        mu=15,
        sigma=5,
        weight=1 / 2
    )

    gaussian4 = gaussians.Gaussian(
        mu=-20,
        sigma=10,
        weight=1 / 8
    )

    multi_gaussian = gaussians.MultiGaussian((gaussian1, gaussian2, gaussian3, gaussian4), sde)
    score_func = multi_gaussian.get_score_function()

    # Create reverse sde_lib based on score function
    reverse_sde = sde.get_reverse_sde(score_func)

    # Create data
    n_samples = 100000
    samples = multi_gaussian.sample(n_samples).unsqueeze(-1)
    x = torch.sort(samples)[0]

    x_start = sde.sample(x, torch.Tensor([1]))

    # Create solver iterables
    n_evaluation_points = 100

    step_range = (10, 100)
    em_evaluation_range = torch.round(torch.exp(torch.linspace(math.log(step_range[0]), math.log(step_range[1]), n_evaluation_points)))
    em_constructor = lambda n_steps: solver_lib.EulerMarayumaSolver(reverse_sde, torch.linspace(1, 0, int(n_steps)))

    tolerance_range = (0.1, 1.3)
    pi_evaluation_range = torch.exp(torch.linspace(math.log(tolerance_range[0]), math.log(tolerance_range[1]), n_evaluation_points))

    pi_constructor = lambda tolerance: solver_lib.PISolver(
        reverse_sde,
        ki=0.3,
        kp=0.1,
        tau_a=0.1,
        tau_r=tolerance,
        alpha=0.9,
        h_start=0.01,
        max_decrease=0.7,
        max_increase=1.20,
    )

    em_solvers = create_solvers(em_constructor, em_evaluation_range)
    pi_solvers = create_solvers(pi_constructor, pi_evaluation_range)

    # Evaluate
    df = pd.DataFrame()

    df["pi_nfe"],df["pi_error"] = evaluate_solvers(pi_solvers, x_start, x, reverse_sde, seed)
    df["em_nfe"],df["em_error"] = evaluate_solvers(em_solvers, x_start, x, reverse_sde, seed)
    df["pi_tau"]                = pi_evaluation_range

    # Write data
    df.to_csv(write_path + "data.csv")

    # Create plot
    plt.figure()
    sns.scatterplot(df, x="em_nfe", y="em_error", label="em")
    sns.scatterplot(df, x="pi_nfe", y="pi_error", label="pi")
    plt.xlabel("NFE")
    plt.xlim(0, 100)
    # plt.ylim(0, 0.2)
    plt.ylabel("Error")
    plt.title("NFE - Error tradeoff of EM and PI solver")
    plt.legend()
    plt.savefig(write_path + "plot.png")
    plt.show()

