import pickle
import os
import torch
import PIL.Image
import pandas as pd
import tqdm

import dnnlib
from sde_lib import EDMSDE, SDE
from solver_lib import *


class PIDataLogger:

    def __init__(self, write_path: str, max_iter: int = 1000, batch_size: int = 64, device: torch.device | str = "cpu"):
        self._write_path = write_path

        self._batch_size = batch_size
        self._ts = torch.zeros(batch_size, max_iter).to(device)
        self._hs = torch.zeros(batch_size, max_iter).to(device)
        self._errors = torch.zeros(batch_size, max_iter).to(device)

        self._i = 0
        self._device = device

    def __call__(self, x: torch.Tensor, t: torch.Tensor, h: torch.Tensor, error: torch.Tensor):
        return self.callback(x, t, h, error)

    def callback(self, x: torch.Tensor, t: torch.Tensor, h: torch.Tensor, error: torch.Tensor):
        print(f"\r{self._i}", end="")

        batch_size = x.shape[0]

        self._ts[:batch_size, self._i] = t.to(self._device).reshape(batch_size)
        self._hs[:batch_size, self._i] = h.to(self._device).reshape(batch_size)
        self._errors [:batch_size, self._i] = error.to(self._device).reshape(batch_size)

        self._i += 1

    def write(self):
        os.makedirs(self._write_path, exist_ok=True)
        self._i = 0

        ts, hs, errors = pd.DataFrame(self._ts.numpy()), pd.DataFrame(self._hs.numpy()), pd.DataFrame(self._errors.numpy())
        t_path, h_path, error_path = self._write_path + "_t.csv", self._write_path + "_h.csv", self._write_path + "_error.csv"

        ts.to_csv(t_path, mode="a", header=False if os.path.exists(t_path) else True)
        hs.to_csv(h_path, mode="a", header=False if os.path.exists(h_path) else True)
        errors.to_csv(error_path, mode="a", header=False if os.path.exists(error_path) else True)

        self._ts = torch.zeros_like(self._ts)
        self._hs = torch.zeros_like(self._hs)
        self._errors = torch.zeros_like(self._errors)


###
# From https://github.com/NVlabs/edm2/blob/main/generate_images.py
class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])
###


def generate_images(
        solver_func: Callable[[SDE], [Solver]],
        outdir: str,
        seed: int = 0,
        n_samples: int = 50000,
        batch_size: int = 64,
        ode_threshold: float = 0.2,
        ode: bool = True,
        model_url: str = "../model/edm2-img64-xl-0671088-0.040.pkl",
        device: torch.device | str = "cuda",
        callback: PIDataLogger | None = None
):
    os.makedirs(outdir)
    seeds = range(seed, n_samples + seed)

    # Load model
    with dnnlib.util.open_url(model_url) as f:
        data = pickle.load(f)
    model = data["ema"].to(device)

    # Load encoder
    encoder = data.get('encoder', None)
    if encoder is None:
        encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')

    sde_ = EDMSDE(ode=ode).to(device).get_reverse_sde(model, ode_threshold=ode_threshold)
    solver = solver_func(sde_).to(device)

    nfe = 0

    # Sampling loop
    for i in tqdm.tqdm(range(0, n_samples, batch_size)):
        # Bound batch size if new batch would exceed total amount of samples
        if (n_samples - i) < batch_size:
            batch_size = n_samples - i

        # Get seeds for the batch
        batch_seeds = seeds[i:(i + batch_size)]

        # Get noise and labels
        rng = StackedRandomGenerator(device, batch_seeds)
        noise = rng.randn((batch_size, model.img_channels, model.img_resolution, model.img_resolution), device=device) * 80
        labels = torch.eye(model.label_dim, device=device)[rng.randint(model.label_dim, size=[len(batch_seeds)], device=device)]

        # Sample using generated noise
        images = solver.solve(noise, labels, callback)

        # Save images
        for seed, image, label in zip(batch_seeds, encoder.decode(images).permute(0, 2, 3, 1).cpu().numpy(), labels):
            label = torch.argmax(label)
            PIL.Image.fromarray(image, "RGB").save(os.path.join(outdir, f"{seed:06d}-{label}.png"))

        if callback is not None:
            callback.write()

        nfe += sde_.nfe
        sde_.reset()

    return nfe / n_samples


def get_pi_solver_func(max_iter: int) -> Callable[[SDE], Solver]:
    return lambda sde_: PISolver(
        sde_,
        ki=0.3,
        kp=0.1,
        tau_a=0.289,
        tau_r=1.95,
        alpha=0.9,
        h_start=12,
        max_decrease=0.05,
        max_increase=2,
        max_iter=max_iter,
        interval=(80, 0),
    )


def get_em_solver_func(nfe: int, t_min: float = 0.002, t_max: float = 80, rho: float = 7):
    return lambda sde: EulerMarayumaSolver(
        sde,
        get_edm_schedule(nfe, t_min, t_max, rho)
    )


def get_heun_solver_func(nfe: int, t_min: float = 0.002, t_max: float = 80, rho: float = 7):
    return lambda sde: HeunSolver(
        sde,
        get_edm_schedule(nfe // 2, t_min, t_max, rho)
    )


if __name__ == "__main__":
    t_min, t_max = 0.002, 80
    n_steps = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 48
    n_samples = 10000
    max_iter = 150

    image_out_path = "../data/image_testing/heun/100NFE/images/"
    data_out_path = "../data/image_testing/heun/100NFE/data/"

    # logger = PIDataLogger(data_out_path, batch_size=batch_size, max_iter=max_iter)
    constructor = get_heun_solver_func(n_steps)

    nfe = generate_images(
        solver_func=constructor,
        outdir=image_out_path,
        n_samples=n_samples,
        batch_size=batch_size,
        device=device,
        ode=False,
        callback=None
    )
    print(nfe)

    os.makedirs(data_out_path, exist_ok=True)
    with open(data_out_path + "nfe.txt", "w") as f:
        f.write("nfe = " + str(nfe))
