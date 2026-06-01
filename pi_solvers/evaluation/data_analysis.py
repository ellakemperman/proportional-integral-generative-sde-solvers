import pandas as pd
import numpy as np
from KDEpy import FFTKDE
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm


def assign_bin(x: float, bins: np.array):
    hist = np.histogram(x, bins=bins)[0]
    if np.max(hist) == 0:
        return -1
    return np.argmax(hist)


def analyse_pi_data(data_path: str, t_max: float = 80, t_min: float = 0.2, plot_res: int = 200):
    # Load in t csv
    ts = pd.read_csv(data_path + "/_t.csv").to_numpy()
    # Add start time to histogram
    ts[:, 0] = np.full(ts.shape[0], t_max)

    # Load in h csv
    hs = pd.read_csv(data_path + "/_h.csv").to_numpy()
    hs[:, 0] = np.full(hs.shape[0], ts[0, 1] - t_max)

    # Flatten and emove 0 values from ts and hs
    ts = ts.reshape(-1)
    hs = hs.reshape(-1)
    mask = ts != 0
    ts = ts[mask]
    hs = hs[mask]

    # Take log of time
    ts = np.log(ts)
    hs = np.log(np.abs(hs))

    print("Plotting...")
    # Plotting
    plt.figure(figsize=(15, 7))
    plt.title("Time conditional probability density of step size.")

    mesh = plt.hexbin(ts, hs, cmap="inferno", gridsize=(plot_res, plot_res), mincnt=0, bins="log")
    plt.colorbar(mesh, label='Probability Density')

    plt.xlim(np.log(t_max), np.log(t_min))
    # plt.xscale("log")
    plt.xlabel("noise/t")
    plt.ylabel("h")

    plt.show()


if __name__ == "__main__":
    analyse_pi_data("../../data/image_testing/pi_2/75NFE_2/data", t_min=0.05, plot_res=100)
