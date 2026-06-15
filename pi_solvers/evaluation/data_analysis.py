import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from pi_solvers.solver_lib import get_edm_schedule
from pi_solvers.utils import compute_discretisation_interpolation


plt.style.use("science")


def detect_outliers(paths: np.ndarray, t_min: float = 0.05, n_outlier_steps: int = 15):
    lengths = []

    for i, path in enumerate(paths):
        end_index = np.where(path <= t_min)[0][0]
        path = path[:(end_index + 1)]

        if end_index <= 2:
            continue

        if path.shape[0] < n_outlier_steps:
            print(f"Outlier at {i} with lengths {path.shape[0]}")

        lengths.append(path.shape[0])

    plt.figure()
    plt.hist(lengths)
    plt.show()



def get_paths(data_path: str, t_max: float, t_min: float, plot_res: int):
    # Load in t csv
    ts = pd.read_csv(data_path + "/_t.csv").to_numpy()
    # Add start time to histogram
    ts[:, 0] = np.full(ts.shape[0], t_max)

    t_grid, paths = compute_discretisation_interpolation(ts, plot_res, t_min=t_min)

    means, stds = paths.mean(axis=0), paths.std(axis=0)

    t_grid = np.linspace(0, 1, means.shape[0])

    return t_grid, means, stds, paths


def generate_pi_image_trajectories(ax, label: str, data_path: str, t_max: float = 80, t_min: float = 0.05, plot_res: int = 200, color: str = "r", n_paths: int = 3, seed=0):
    np.random.seed(seed)
    t_grid, means, stds, paths = get_paths(data_path, t_max, t_min, plot_res)

    random_paths = paths[np.random.randint(0, paths.shape[0], n_paths), :]

    # EDM discretisation
    discretisation = get_edm_schedule(plot_res, t_min=t_min)[:-1]

    print("Plotting...")
    # Plotting
    ax.plot(t_grid, means, label=label, c=color)
    ax.fill_between(t_grid, means + stds, means -stds, alpha=0.1, color=color)

    # Plot random paths
    for i in range(n_paths):
        ax.plot(t_grid, random_paths[i, :], label=f"Sample Path {i}", linewidth=1)

    plt.plot(np.linspace(0, 1, discretisation.shape[0]), discretisation, label="EDM Discretisation")
    ax.legend()
    plt.yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction along SDE path")
    ax.set_ylim(t_min, t_max)
    ax.set_ylabel(r"$\sigma$")
    ax.grid()
    return ax


def analyse_pi_gaussian_trajectories(ax, label: str, data_path: str, t_max: float = 1, t_min: float = 0, plot_res: int = 200, color: str = "r", n_paths: int = 3, seed=0):
    np.random.seed(seed)
    t_grid, means, stds, paths = get_paths(data_path, t_max, t_min, plot_res)

    random_paths = paths[np.random.choice(paths.shape[0], n_paths), :]

    print("Plotting...")
    # Plotting
    ax.plot(t_grid, means, label=label, c=color)
    ax.fill_between(t_grid, means - stds, means + stds, alpha=0.1, color=color)

    # Plot random paths
    for i in range(n_paths):
        ax.plot(t_grid, random_paths[i, :], label=f"Sample Path {i}", linewidth=1)

    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction along SDE path")
    ax.set_ylim(t_min, t_max)
    ax.set_ylabel(r"$t$")
    ax.grid()
    return ax


if __name__ == "__main__":
    data_path = "../../data/gaussian_test/simple_high_h_start"
    t_min = 0.05
    t_max = 80

    fig = plt.figure()
    fig.set_size_inches(5, 3.75)
    ax = fig.add_subplot(111)

    # generate_pi_image_trajectories(ax, data_path=data_path, label="PI Average", t_min=t_min, t_max=t_max, n_paths=2)
    analyse_pi_gaussian_trajectories(ax, data_path=data_path, label="PI Average", t_max=1, t_min=0, n_paths=3)

    fig.show()

    ts = pd.read_csv(data_path + "/_t.csv").to_numpy()
    # Add start time to histogram
    ts[:, 0] = np.full(ts.shape[0], t_max)

    detect_outliers(ts, t_min)