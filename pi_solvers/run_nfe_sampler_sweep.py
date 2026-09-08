from ml_collections import config_dict

import argparse

from pi_solvers.generate_images import *
from pi_solvers.evaluate_images import *
from pi_solvers.utils.utils import load_config


def resolve_spec(cfg: config_dict, spec: config_dict, seed: int = None, out_dir: str = None):
    sampler_name = spec.sampler_name

    # Resolve sampler name
    if sampler_name.lower() in ["euler-maruyama", "em"]:
        sampler_func = generate_em_images

    elif sampler_name.lower() in ["heun", "stochastic_heun"]:
        sampler_func = lambda **kwargs: generate_em_images(heun=True, **kwargs)

    elif sampler_name.lower() in ["edm", "karras"]:
        sampler_func = generate_edm_images

    elif sampler_name.lower() in ["ggf", "gotta-go-fast"]:
        sampler_func = generate_ggf_images

    elif sampler_name.lower() in ["pi", "propotional-integral"]:
        sampler_func = generate_pi_images

    else:
        raise ValueError(f"Sampler {sampler_name} does not exist.")

    # Resolve schedule name
    sampler_schedule_name = spec.sampler_schedule_name

    if sampler_schedule_name.lower() in ["pi", "pi-static", "propotional-integral"]:
        func = lambda **kwargs: sampler_func(pi_discretisation=spec.sampler_schedule_path, **kwargs)
    else:
        func = sampler_func

    # Generate images
    if out_dir is not None:
        out_path = out_dir + spec.spec_path
    else:
        out_path = spec.out_path

    seed = cfg.seed if seed is None else seed

    print(f"Generating {cfg.n_samples} images for the {sampler_name} solver with {sampler_schedule_name} schedule at {spec.nfe} NFE...")
    func(
        batch_size=cfg.sampling_batch_size,
        device=cfg.device,
        n_images=cfg.n_samples,
        model=cfg.model,
        seed=seed,
        output=out_path,
        ode=cfg.ode,
        exist_okay=cfg.exist_okay,
        ode_threshold=spec.ode_threshold,
        nfe=spec.nfe,
        **spec.sampler_kwargs
    )

    # Evaluating images
    metrics = eval_features(
        sample_dir=out_path + "images",
        ref_dir=cfg.feature_path,
        metric=cfg.metrics,
        output=out_path + "data",
        ref_statistics=cfg.stats_path,
        device=cfg.device,
        n_images=cfg.n_samples,
        batch_size=cfg.eval_batch_size
    )

    #
    with open(cfg.metrics_out + "results.txt", "a") as f:
        f.write(f"{sampler_name}-{sampler_schedule_name}-{spec.nfe}: ")
        for metric in metrics:
            f.write(metric + ", ")
            f.write("\n")



def main():
    parser = argparse.ArgumentParser(description="Generates images using diffusion using EDM models, from sigma=80 to sigma=0, using varying samplers.")
    parser.add_argument("config", type=str, help="config file configuring specs and hyperparameters")
    parser.add_argument("--out_dir", "-o", type=str, default=None, help="Output directory override")
    parser.add_argument("--seed", "-s", type=int, default=None, help="Seed override")
    args = parser.parse_args()

    cfg = load_config(args.config)

    for spec in cfg.specs:
        resolve_spec(cfg, spec, args.seed, args.out_path)

    print(f"Sweep done, output can be found at {cfg.metrics_out}/results.txt")


if __name__ == "__main__":
    main()
