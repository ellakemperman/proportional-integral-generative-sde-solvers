from typing import Callable
import os

import torch
import numpy as np
import PIL.Image
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

from pi_solvers.solver_lib import Solver, construct_heun_end_adaptive_solver, PISolver
from pi_solvers.sde_lib import SDE, StableDiffusionVPSDE


class StableDiffusionModel:

    def __init__(
            self,
            prompt: str,
            sde: StableDiffusionVPSDE,
            negative_prompt: str = "",
            guidance_scale: float = 7.5,
            model_name: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
            device: torch.device | str = "cuda"
    ):
        self._device = device
        self._guidance_scale = guidance_scale

        self._tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self._text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(device)
        self._unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").to(device)
        self._vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(device)

        self._embeddings = self._encode(prompt)
        self._negative_embeddings = self._encode(negative_prompt)

        self._sde = sde

    @property
    def network(self) -> UNet2DConditionModel:
        return self._unet

    def _encode(self, text: str):
        tokens = self._tokenizer(text, padding="max_length",
                                 max_length=self._tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt").to(self._device)

        return self._text_encoder(tokens.input_ids)[0]

    def __call__(self, x, t, _):
        # UNet from stable diffusion wants DDPM timesteps (0-999)
        index = self._sde.get_index_from_t(t).reshape(-1)

        # Concatenate for CFG
        x_in = torch.cat([x, x], dim=0).to(torch.float32)
        embeds = torch.cat([self._negative_embeddings, self._embeddings], dim=0)

        # Forward calls
        with torch.no_grad():
            noise_pred = self._unet(x_in, index, encoder_hidden_states=embeds).sample

        # Handle CFG
        unconditional_noise, conditional_noise = noise_pred.chunk(2)
        epsilon = unconditional_noise + self._guidance_scale * (conditional_noise - unconditional_noise)
        return (- epsilon / self._sde.sigma(t)).to(torch.float)

    def decode(self, x: torch.Tensor) -> np.ndarray:
        latents = x / self._vae.config.scaling_factor

        with torch.no_grad():
            images = self._vae.decode(latents).sample

        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()  # -> [B, H, W, 3]
        return (images * 255).round().astype("uint8")


def sample_sd(
        prompt: str,
        solver_func: Callable[[SDE, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]], Solver],
        outdir: str,
        batch_size: int = 1,
        seed: int = 0,
        ode: bool = False,
        ode_threshold: float = 0,
        negative_prompt: str = "",
        guidance_scale: float = 7.5,
        model_name: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        device: str = "cuda"
):
    os.makedirs(outdir, exist_ok=True)

    print("Loading model...")
    sde = StableDiffusionVPSDE(ode=ode, seed=seed).to(device)
    model = StableDiffusionModel(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        model_name=model_name,
        device=device,
        sde=sde
    )

    rng = torch.Generator().manual_seed(seed)

    rsde = sde.get_reverse_sde(model, ode_threshold=ode_threshold).to(device)

    solver = solver_func(rsde, model).to(device)

    res = model.network.config["sample_size"]
    channels = model.network.config["in_channels"]

    def pi_callback(x, t, h, error):
        print(f"t = {torch.mean(t)}", end=" ")
        print(f"h = {torch.mean(h)}", end=" ")
        print(f"error = {torch.mean(error)}")

    x = torch.randn((batch_size, channels, res, res), generator=rng).to(device)
    print(x.shape)
    x_solved = solver.solve(x, callback=pi_callback)
    print(f"nfe: {rsde.nfe / batch_size}")
    images = model.decode(x_solved)
    print(images.shape)

    for i, image in enumerate(images):
        PIL.Image.fromarray(image, "RGB").save(os.path.join(outdir, f"{i}.png"))


if __name__ == "__main__":
    solver_constructor = lambda sde, _: construct_heun_end_adaptive_solver(
        adaptive_solver_class=PISolver,
        sde=sde,
        seed=0,
        ode_threshold=100,
        n_ode_steps=10,
        ki=0.3,
        kp=0.1,
        tau_a=0.06,
        tau_r=2,
        alpha=0.9,
        h_start=10,
        max_decrease=0.02,
        max_increase=5,
        interval=(999, 0),
        abs_error=False,
        batch_norm=False
    )

    sample_sd(
        "god smiting a tiny mortal city to ashes",
        solver_func=solver_constructor,
        outdir="../../data/sd/test/",
        guidance_scale=7.5,
        ode=False,
        batch_size=1,
    )
