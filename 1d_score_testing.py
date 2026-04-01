import pandas as pd
import torch
from scipy import stats
from typing import Iterable, Callable, Any
import math
import solvers
import sde
import gaussians


def evaluate_solvers(solvers: Iterable[solvers.Solver],
                     x_start: torch.Tensor,
                     x: torch.Tensor,
                     sde_: sde.SDE,
                     seed: int = 0
                     ) -> tuple[list[float], list[float]]:
    nfe, distance = [], []
    for solver in solvers:
        print("Hello")
        torch.manual_seed(seed)
        sde_.reset_nfe()
        x_hat = solver.solve(x_start.clone())
        nfe.append(sde_.nfe)
        distance.append(
            stats.wasserstein_distance(x[:, 0], torch.sort(x_hat)[0][:, 0])
        )
    return nfe, distance


def create_solvers(
        constructor: Callable[[Any], solvers.Solver],
        parameter_range: Iterable[Any]
) -> Iterable[solvers.Solver]:
    for param in parameter_range:
        yield constructor(param)


if __name__ == "__main__":
    seed = 42
    write_path = "/results"
    torch.manual_seed(seed)

    # SDE Setup
    beta_min = 0.1
    beta_max = 20
    sde = sde.LinearVariancePreservingSDE(beta_min, beta_max)

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

    # Create reverse sde based on score function
    reverse_sde = sde.get_reverse_sde(score_func)

    # Create data
    n_samples = 1000000
    samples = multi_gaussian.sample(n_samples).unsqueeze(-1)
    x = torch.sort(samples)[0]

    x_start = sde.sample(x, torch.Tensor([1]))

    # Create solver iterables
    n_evaluation_points = 3

    step_range = (1, 1000)
    em_evaluation_range = torch.round(torch.exp(torch.linspace(math.log(step_range[0]), math.log(step_range[1]), n_evaluation_points)))
    em_constructor = lambda n_steps: solvers.EulerMarayumaSolver(reverse_sde, torch.linspace(1, 0, int(n_steps)))

    tolerance_range = (0.5, 2)
    pi_evaluation_range = torch.round(torch.exp(torch.linspace(math.log(0.1), math.log(10), n_evaluation_points)))
    pi_constructor = lambda tolerance: solvers.PISolver(
        reverse_sde,
        ki=0.101,
        kp=0.09,
        tau=tolerance,
        alpha=0.5,
        h_start=0.01,
        max_decrease=0.7,
        max_increase=1.15
    )

    em_solvers = create_solvers(em_constructor, em_evaluation_range)
    pi_solvers = create_solvers(pi_constructor, pi_evaluation_range)

    # Evaluate
    df = pd.DataFrame()

    df["em_nfe"],df["em_error"] = evaluate_solvers(em_solvers, x_start, x, reverse_sde, seed)
    df["pi_nfe"],df["pi_error"] = evaluate_solvers(pi_solvers, x_start, x, reverse_sde, seed)

    # Write data
    df.to_csv(write_path)
