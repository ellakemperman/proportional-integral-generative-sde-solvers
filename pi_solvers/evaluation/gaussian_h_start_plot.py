import scienceplots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


if __name__ == "__main__":
    path = "../../data/gaussian_test/grid/h_start_complex"
    hs = [0.01, 0.05, 0.1, 0.15, 0.2]


    plt.figure(figsize=(8, 5))
    plt.style.use("science")
    for h in hs:
        data = pd.read_csv(path + "/" + str(h) + "/data.csv")
        x = data["pi_nfe"]
        y = data["pi_error"]
        y = np.convolve(y, np.array([0.25, 0.5, 1, 0.5, 0.25]))[2:-2]

        plt.plot(x, y, label=fr"$h_0={h}$", linewidth=1.5)

    plt.xlim(0, 100)
    plt.xlabel("NFE")
    plt.ylabel(r"$D_W$")
    plt.yscale("log")
    plt.legend()
    plt.show()

