import pickle
import PIL
import os

import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import sde_lib
from solver_lib import PISolver, PISolver2
import dnnlib


def create_grid(tau_a_range: tuple[float, float], tau_r_range: tuple[float, float], resolution: int) -> tuple[
    torch.Tensor, torch.Tensor]:
    tau_as = torch.linspace(tau_a_range[0], tau_a_range[1], resolution)
    tau_rs = torch.linspace(tau_r_range[0], tau_r_range[1], resolution)
    av, rv = torch.meshgrid((tau_as, tau_rs))
    return av, rv


class RejectCounter:
    def __init__(self):
        self.__reject_count = 0
        self.__total = 0

    def __call__(self, x: torch.Tensor, t: torch.Tensor, h: torch.Tensor, error: torch.Tensor):
        return self.callback(x, t, h, error)

    def callback(self, x: torch.Tensor, t: torch.Tensor, h: torch.Tensor, error: torch.Tensor):
        self.__total += error.shape[0]
        self.__reject_count += int(torch.sum(error > 1))

    def reject_rate(self):
        return self.__reject_count / self.__total


def apply_over_grid(
        noise: torch.Tensor,
        labels: torch.Tensor,
        sde: sde_lib.SDE,
        seed: int,
        grid: tuple[torch.Tensor, torch.Tensor],
        device: torch.device | str,
        encoder,
        outdir: str,
        ode_threshold: float,
        n_ode_steps: int,
        ki: float,
        kp: float,
        alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    nfes = torch.zeros_like(grid[0])
    reject_rate = torch.zeros_like(grid[0])

    for i in tqdm(range(grid[0].shape[1])):
        for j in tqdm(range(grid[0].shape[0])):
            torch.random.manual_seed(seed)

            tau_a, tau_r = float(grid[0][i, j]), float(grid[1][i, j])
            sde.reset()

            reject_counter = RejectCounter()

            solver = PISolver2.create_heun_end_pi_solver(
                rsde,
                ode_threshold=ode_threshold,
                n_ode_steps=n_ode_steps,
                ki=ki,
                kp=kp,
                tau_a=tau_a,
                tau_r=tau_r,
                alpha=alpha,
                h_start=0.5,
                max_decrease=0.05,
                max_increase=10,
                max_iter=1000,
                interval=(80, 0.002),
                abs_error=False
            ).to(device)

            images = solver.solve(noise.clone(), labels=labels, callback=reject_counter)

            for k, (image, label) in enumerate(zip(encoder.decode(images).permute(0, 2, 3, 1).cpu().numpy(), labels)):
                label = torch.argmax(label)
                dir_path = os.path.join(outdir, f"abs_{round(tau_a, 3)}_rel_{round(tau_r, 3)}")
                os.makedirs(dir_path, exist_ok=True)
                PIL.Image.fromarray(image, "RGB").save(os.path.join(dir_path, f"{k}_{label}.png"))

            nfes[i, j] = sde.nfe
            reject_rate[i, j] = reject_counter.reject_rate()

    return nfes, reject_rate

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Out path
    outdir = "data/tolerance_grid/eval_pisolver_2"
    os.makedirs(outdir, exist_ok=True)

    # hyper parameters
    batch_size = 64
    tau_a_range = (0.05, 0.5)
    tau_r_range = (0.5, 10)
    resolution = 20
    ki = 0.3
    kp = 0.1
    alpha = 0.9
    ode_threshold = 0.2
    n_ode_steps = 3
    checkpoint = "model/edm2-img64-xl-0671088-0.040.pkl"
    seed = 42

    # Set seed
    torch.random.manual_seed(seed)

    # Load model
    with dnnlib.util.open_url(checkpoint) as f:
        data = pickle.load(f)
    model = data["ema"].to(device)

    # Create SDE
    sde_ = sde_lib.EDMSDE().to(device)
    rsde = sde_.get_reverse_sde(model).to(device)

    # Sample noise and labels
    x = torch.zeros((batch_size, model.img_channels, model.img_resolution, model.img_resolution)).to(device)
    noise = torch.randn_like(x) * 80
    labels = torch.eye(model.label_dim, device=device)[
        torch.randint(high=model.label_dim, size=(batch_size,), device=device)]

    # Write hyperparameters to info file:
    with open(outdir + "/info.txt", "w") as f:
        f.write(f"batch_size = {batch_size}\n")
        f.write(f"tau_a_range = {tau_a_range}\n")
        f.write(f"tau_r_range = {tau_r_range}\n")
        f.write(f"resolution = {resolution}\n")
        f.write(f"ki = {ki}\n")
        f.write(f"kp = {kp}\n")
        f.write(f"alpha = {alpha}\n")
        f.write(f"ode_threshold = {ode_threshold}\n")
        f.write(f"n_ode_steps = {n_ode_steps}\n")
        f.write(f"checkpoint = {checkpoint}\n")
        f.write(f"labels = {[float(torch.argmax(label)) for label in labels]}")

    # Create evaluation grid
    grid = create_grid(tau_a_range, tau_r_range, resolution)

    # Run the solver over the grid
    nfes, reject_rate = apply_over_grid(
        noise=noise,
        labels=labels,
        sde=rsde,
        seed=seed,
        grid=grid,
        device=device,
        encoder=data.encoder,
        outdir=outdir,
        ode_threshold=ode_threshold,
        n_ode_steps=n_ode_steps,
        ki=ki,
        kp=kp,
        alpha=alpha
    )

    # Save data as CSV
    df_nfe = pd.DataFrame((nfes / batch_size).cpu().numpy())
    df_nfe.to_csv(outdir + "/nfe.csv")

    df_reject = pd.DataFrame(reject_rate.cpu().numpy())
    df_reject.to_csv(outdir + "/reject.csv")

    # Create and save plots
    plt.figure()
    mesh = plt.pcolormesh(grid[0], grid[1], nfes / batch_size, cmap='inferno')
    plt.colorbar(mesh, label='NFE')
    plt.xlabel(r"$\tau_a$")
    plt.ylabel(r"$\tau_r$")
    plt.title(r"NFE as a function of $\tau_a$ and $\tau_r$")
    plt.savefig(outdir + "/nfe.png")

    plt.figure()
    mesh = plt.pcolormesh(grid[0], grid[1], reject_rate, cmap='inferno')
    plt.colorbar(mesh, label='reject rate')
    plt.xlabel(r"$\tau_a$")
    plt.ylabel(r"$\tau_r$")
    plt.title(r"Rejection rate as a function of $\tau_a$ and $\tau_r$")
    plt.savefig(outdir + "/reject.png")
