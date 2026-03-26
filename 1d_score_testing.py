import matplotlib.pyplot as plt
import torch
import solver
import sde
import gaussians


if __name__ == "__main__":
    # Setup
    # SDE Setup
    beta_min = 0.1
    beta_max = 20
    sde = sde.LinearVariancePreservingSDE(beta_min, beta_max)

    # Gaussian & score setup
    gaussian1 = gaussians.Gaussian(
        mu=2,
        sigma=1,
        weight=0.5
    )
    gaussian2 = gaussians.Gaussian(
        mu=-2,
        sigma=1,
        weight=0.5
    )

    multi_gaussian = gaussians.MultiGaussian((gaussian1, gaussian2), sde)
    score_func = multi_gaussian.get_score_function()

    reverse_sde = sde.get_reverse_sde(score_func)

    # Solver setup
    n_steps = 100
    discretisation = torch.linspace(1, 0, n_steps)
    solver = solver.EulerMarayumaSolver(reverse_sde, discretisation)

    # Results gathering
    n_samples = 100000
    mu, sigma = 0, 1

    mu, sigma = torch.full((n_samples,), mu), torch.full((n_samples,), sigma)
    x_start = torch.randn(n_samples)
    x = solver.solve(x_start)

    plt.figure()
    plt.hist(x, bins=100, density=True)
    plt.show()
