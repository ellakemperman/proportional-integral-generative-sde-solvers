import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import numpy as np
import scipy
import pickle
import solvers
import sde
import dnnlib

if __name__ == "__main__":
    torch.manual_seed(42)

    with dnnlib.util.open_url("model/edm2-img64-m-2147483-0.060.pkl") as f:
        model = pickle.load(f)["ema"].to("cuda")

    noise = torch.randn((8, model.img_channels, model.img_resolution, model.img_resolution)).to("cuda")

    sde_ = sde.EDMSDE(0.002, 80).to("cuda")
    # sde_ = sde.VarianceExplodingSDE(sigma_min=0.002, sigma_max=80).to("cuda")
    rsde = sde_.get_reverse_sde(model).to("cuda")

    n_steps = 500
    discretisation = torch.linspace(1, 0, n_steps)

    em_solver = solvers.EulerMarayumaSolver(rsde, discretisation).to("cuda")
    """    pi_solver = solvers.PISolver(
        rsde,
        ki=0.101,
        kp=0.09,
        tau=0.42668694,
        alpha=0.8,
        h_start=0.01,
        max_decrease=0.7,
        max_increase=1.20
    ).to("cuda")"""

    x_pi = em_solver.solve(noise.clone())

    plt.imshow(x_pi[0, 0:3, :, :].cpu().T)