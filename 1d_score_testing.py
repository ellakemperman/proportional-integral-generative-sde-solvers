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
        mu=0,
        sigma=1,
        weight=1/3
    )
    gaussian2 = gaussians.Gaussian(
        mu=-5,
        sigma=1,
        weight=1/3
    )

    gaussian3 = gaussians.Gaussian(
        mu=5,
        sigma=1,
        weight=1/3
    )

    multi_gaussian = gaussians.MultiGaussian((gaussian1, gaussian2, gaussian3), sde)
    score_func = multi_gaussian.get_score_function()

    # Computing function over interval
    interval = torch.linspace(-10, 10, 10000)
    verification = multi_gaussian(interval)

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

    # Plotting
    plt.figure()
    plt.hist(x, bins=100, density=True)
    plt.plot(interval, verification, c="r")
    plt.show()
