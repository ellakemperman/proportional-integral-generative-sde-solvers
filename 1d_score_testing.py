import matplotlib.pyplot as plt
import seaborn as sns
import torch
import solver
import sde
import gaussians


if __name__ == "__main__":
    torch.manual_seed(42)

    # Setup
    # SDE Setup
    beta_min = 0.1
    beta_max = 20
    sde = sde.LinearVariancePreservingSDE(beta_min, beta_max)

    # Gaussian & score setup
    gaussian1 = gaussians.Gaussian(
        mu=0,
        sigma=1,
        weight=1/8
    )
    gaussian2 = gaussians.Gaussian(
        mu=-5,
        sigma=1,
        weight=3/8
    )

    gaussian3 = gaussians.Gaussian(
        mu=15,
        sigma=1,
        weight=1/2
    )

    multi_gaussian = gaussians.MultiGaussian((gaussian1, gaussian2, gaussian3), sde)
    score_func = multi_gaussian.get_score_function()

    # Computing function over interval
    interval = torch.linspace(-10, 20, 10000).reshape((10000, 1))
    verification = multi_gaussian(interval)

    # Solver setup
    reverse_sde = sde.get_reverse_sde(score_func)

    n_steps = 1000
    discretisation = torch.linspace(1, 0, n_steps)
    # solver_ = solver.EulerMarayumaSolver(reverse_sde, discretisation)
    solver_ = solver.PISolver(reverse_sde, ki=0.101, kp=0.09, tau=0.1, alpha=0.8, h_start=0.001, max_decrease=0.7, max_increase=1.3)

    # Results gathering
    n_samples = 1000000
    mu, sigma = 0, 1

    mu, sigma = torch.full((n_samples,), mu), torch.full((n_samples,), sigma)
    x_start = torch.randn((n_samples, 1))
    plt.figure()
    sns.kdeplot(x_start)
    plt.title("Data before transformation")
    plt.show()

    x = solver_.solve(x_start)

    # Plotting
    plt.figure()
    sns.kdeplot(x, label="Sampled", multiple="stack")
    plt.plot(interval, verification, c="r", label="True Multimodal Gaussian")
    plt.legend()
    plt.title("PI Adaptive solver transforming to a multimodal Gaussian")
    plt.show()

    print(multi_gaussian.nfe)
