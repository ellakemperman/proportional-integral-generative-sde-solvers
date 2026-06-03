import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pi_solvers.solver_lib import get_edm_schedule
from pi_solvers.utils import compute_discretisation_interpolation


def assign_bin(x: float, bins: np.array):
    hist = np.histogram(x, bins=bins)[0]
    if np.max(hist) == 0:
        return -1
    return np.argmax(hist)


def analyse_pi_data(data_path: str, t_max: float = 80, t_min: float = 0.05, plot_res: int = 200):
    # Load in t csv
    ts = pd.read_csv(data_path + "/_t.csv").to_numpy()
    # Add start time to histogram
    ts[:, 0] = np.full(ts.shape[0], t_max)

    t_grid, means, stds = compute_discretisation_interpolation(ts, plot_res)

    # Reverse means and stds
    means = means[::-1]
    stds = stds[::-1]

    # Filter out the low times
    mask = t_grid > t_min
    means = means[mask]
    stds = stds[mask]
    t_grid = np.linspace(0, 1, means.shape[0])

    # EDM discretisation
    discretisation = get_edm_schedule(plot_res, t_min=t_min)[:-1]

    print("Plotting...")
    # Plotting
    # plt.figure(figsize=(15, 7))
    # plt.title(r"$\sigma$ as a function of $t$")
    plt.plot(t_grid, means, label="PI Mean Discretisation")
    plt.fill_between(t_grid, means - stds, means + stds, label="±1 std", alpha=0.3)
    plt.plot(np.linspace(1, 0, discretisation.shape[0]), discretisation, label="EDM Discretisation")
    plt.legend()
    plt.yscale("log")
    plt.xlim(1, 0)
    plt.xlabel("fraction along SDE path")
    plt.ylabel("sigma")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    analyse_pi_data("../../data/image_testing/pi_2/75NFE_2/data", t_max=80, t_min=0.05)
