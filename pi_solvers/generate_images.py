import argparse
import datetime
import os

from pi_solvers.solver_lib import *
from pi_solvers.evaluation.generate_samples import generate_images
from pi_solvers.utils.data_logger import PIDataLogger


def setup_dirs(name: str, path: str = None, exist_okay: bool = False) -> tuple[str, str]:
    if path is None:
        path = f"data/image_testing/{name}/{hash(datetime.time())}"
    image_path = path + "/images/"
    write_path = path + "/data/"

    os.makedirs(image_path, exist_ok=exist_okay)
    os.makedirs(write_path, exist_ok=exist_okay)

    return image_path, write_path


def write_general_info(path: str, **kwargs):
    with open(path, "w") as f:
        f.write(str(kwargs))


def generate_pi_images(
        batch_size: int,
        device: torch.device,
        n_images: int,
        model: str,
        seed: int,
        output: str,
        ode: bool,
        exist_okay: bool,
        **pi_kwargs
):
    print(f"Setting up PI-solver for {n_images} images...")
    image_path, write_path = setup_dirs("pi_2", output, exist_okay)

    write_general_info(
        path=write_path + "info.txt",
        batch_size=batch_size,
        device=device,
        n_images=n_images,
        model=model,
        seed=seed,
        ode=ode,
        exist_okay=exist_okay,
        **pi_kwargs
    )

    solver_constructor = lambda sde, _: PISolver2.create_heun_end_pi_solver(sde=sde, **pi_kwargs)
    nfe = generate_images(
        solver_func=solver_constructor,
        outdir=image_path,
        n_samples=n_images,
        batch_size=batch_size,
        ode=ode,
        seed=seed,
        model_url=model,
        device=device,
        callback=PIDataLogger(write_path=write_path, max_iter=pi_kwargs["max_iter"], batch_size=batch_size)
    )

    with open(write_path + "info.txt", 'a') as f:
        f.write(f"nfe: {nfe}")


def generate_em_images(
        batch_size: int,
        device: torch.device,
        n_images: int,
        model: str,
        seed: int,
        output: str,
        ode: bool,
        exist_okay: bool,
        nfe: int,
        rho: float
):
    print(f"Setting up EM-solver for {n_images} images...")
    image_path, write_path = setup_dirs("em", output, exist_okay)

    write_general_info(
        path=write_path + "info.txt",
        batch_size=batch_size,
        device=device,
        n_images=n_images,
        model=model,
        seed=seed,
        ode=ode,
        exist_okay=exist_okay,
        nfe=nfe,
        rho=rho
    )

    discretisation = get_edm_schedule(nfe, rho=rho)
    solver_constructor = lambda sde, _: EulerMarayumaSolver(sde=sde, discretisation=discretisation)
    generate_images(
        solver_func=solver_constructor,
        outdir=image_path,
        n_samples=n_images,
        batch_size=batch_size,
        ode=ode,
        seed=seed,
        model_url=model,
        device=device
    )


def generate_edm_images(
        batch_size: int,
        device: torch.device,
        n_images: int,
        model: str,
        seed: int,
        output: str,
        ode: bool,
        exist_okay: bool,
        nfe: int,
        rho: float,
        **edm_kwargs
):
    print(f"Setting up EDM-solver for {n_images} images...")
    image_path, write_path = setup_dirs("edm", output, exist_okay)

    write_general_info(
        path=write_path + "info.txt",
        batch_size=batch_size,
        device=device,
        n_images=n_images,
        model=model,
        seed=seed,
        ode=ode,
        exist_okay=exist_okay,
        nfe=nfe,
        rho=rho,
        **edm_kwargs
    )

    discretisation = get_edm_schedule(nfe // 2, rho=rho)
    solver_constructor = lambda _, model: EDMSolver(model=model, discretisation=discretisation, **edm_kwargs)
    generate_images(
        solver_func=solver_constructor,
        outdir=image_path,
        n_samples=n_images,
        batch_size=batch_size,
        ode=ode,
        seed=seed,
        model_url=model,
        device=device
    )


def main():
    parser = argparse.ArgumentParser(description="Generates images using diffusion")
    parser.add_argument("-b", "--batch_size", default=64, type=int)
    parser.add_argument("-d", "--device", default="cuda", type=torch.device)
    parser.add_argument("-n", "--n_images", default=0, type=int)
    parser.add_argument("--ode", action='store_true')
    parser.add_argument("-m", "--model", default="model/edm2-img64-xl-0671088-0.040.pkl", type=str)
    parser.add_argument("-s", "--seed", default=0, type=int)
    parser.add_argument("-o", "--output", default=None, type=str)
    parser.add_argument("-e", "--exist_okay", action='store_true')

    subparsers = parser.add_subparsers()

    em_parser = subparsers.add_parser("euler-marayuma", aliases=["em"])
    em_parser.add_argument("--nfe", default=50, type=int)
    em_parser.add_argument("--rho", default=7, type=float)
    em_parser.set_defaults(func=generate_em_images)

    edm_parser = subparsers.add_parser("edm")
    edm_parser.add_argument("--nfe", default=50, type=int)
    edm_parser.add_argument("--rho", default=7, type=float)
    edm_parser.add_argument("--S_churn", default=0, type=float)
    edm_parser.add_argument("--S_min", default=0, type=float)
    edm_parser.add_argument("--S_max", default=float("inf"), type=float)
    edm_parser.add_argument("--S_noise", default=1, type=float)
    edm_parser.set_defaults(func=generate_edm_images)

    pi_parser = subparsers.add_parser("proportional-integral", aliases=["pi"])
    pi_parser.add_argument("--max_iter", default=1000, type=int)
    pi_parser.add_argument("--ode_threshold", default=0.2, type=float)
    pi_parser.add_argument("--n_ode_steps", default=3, type=int)
    pi_parser.add_argument("--ki", default=0.3, type=float)
    pi_parser.add_argument("--kp", default=0.1, type=float)
    pi_parser.add_argument("--tau_a", default=0.1, type=float)
    pi_parser.add_argument("--tau_r", default=10, type=float)
    pi_parser.add_argument("--alpha", default=0.9, type=float)
    pi_parser.add_argument("--h_start", default=30, type=float)
    pi_parser.add_argument("--max_decrease", default=0.05, type=float)
    pi_parser.add_argument("--max_increase", default=5, type=float)
    pi_parser.add_argument("--batch_norm", action='store_true')
    pi_parser.add_argument("--abs_error", action='store_true')
    parser.set_defaults(func=generate_pi_images)

    args = parser.parse_args()
    args.func(**vars(args))

    print("Finished")


if __name__ == "__main__":
    main()
