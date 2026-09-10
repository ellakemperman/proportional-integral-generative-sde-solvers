from ml_collections import config_dict

from pi_solvers.evaluation.metrics import Metrics


def get_config():
    cfg = config_dict.ConfigDict()

    # Sampling
    cfg.model = "https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-xl-0671088-0.040.pkl"
    cfg.n_samples = 50000
    cfg.sampling_batch_size = 128
    cfg.device = "cuda:0"
    cfg.seed = 0
    cfg.ode = False
    cfg.exist_okay = True
    cfg.base_out = "data/imagenet-64/"

    # Evaluation
    cfg.metrics = [Metrics.FID]
    cfg.feature_path = "refs/img_features.pkl"
    cfg.stats_path = "refs/img_stats.pkl"
    cfg.eval_batch_size = 1024
    cfg.metrics_out = cfg.base_out

    cfg.specs = []

    def add_spec(
            sampler_name: str,
            sampler_schedule_name: str,
            nfe: int,
            ode_threshold: float = 0.05,
            sampler_schedule_path: str | None = None,
            **sampler_kwargs
    ):
        spec = config_dict.ConfigDict()
        spec.sampler_name = sampler_name
        spec.sampler_schedule_name = sampler_schedule_name
        spec.nfe = nfe
        spec.sampler_schedule_path = sampler_schedule_path
        spec.sampler_kwargs = sampler_kwargs
        spec.out_path = f"{cfg.base_out}/{sampler_name}_{sampler_schedule_name}/{nfe}NFE/"
        spec.spec_path = f"{sampler_name}_{sampler_schedule_name}/{nfe}NFE/"
        spec.ode_threshold = ode_threshold
        cfg.specs.append(spec)

    # PI specs
    pi_params = lambda tau_rel, h_0, n: {
        "n_ode_steps": n,
        "ki": 0.3,
        "kp": 0.1,
        "tau_a": 0.005,
        "tau_r": tau_rel,
        "alpha": 0.9,
        "h_start": h_0,
        "max_decrease": 0.2,
        "max_increase": 5,
        "max_iter": 1000
    }

    add_spec("pi", "adaptive", 49, **pi_params(1.08, 45, 5))
    add_spec("pi", "adaptive", 75, **pi_params(0.87, 40, 7))
    add_spec("pi", "adaptive", 99, **pi_params(0.72, 35, 10))

    # PI specs
    pi_params = lambda tau_rel, h_0, n: {
        "n_ode_steps": n,
        "ki": 0.3,
        "kp": 0,
        "tau_a": 0.005,
        "tau_r": tau_rel,
        "alpha": 0.9,
        "h_start": h_0,
        "max_decrease": 0.2,
        "max_increase": 5,
        "max_iter": 1000
    }

    add_spec("pi_integral", "adaptive", 49, **pi_params(1.015, 45, 5))
    add_spec("pi_integral", "adaptive", 75, **pi_params(0.83, 40, 7))
    add_spec("pi_integral", "adaptive", 99, **pi_params(0.7, 35, 10))

    # Stochastic Heun
    add_spec("heun", "pi", 49, sampler_schedule_path=f"{cfg.base_out}/pi_adaptive/49NFE/data/_t.csv")
    add_spec("heun", "pi", 75, sampler_schedule_path=f"{cfg.base_out}/pi_adaptive/75NFE/data/_t.csv")
    add_spec("heun", "pi", 99, sampler_schedule_path=f"{cfg.base_out}/pi_adaptive/99NFE/data/_t.csv")

    # EDM
    edm_churn_params = {
        "S_churn": 40,
        "S_min": 0.05,
        "S_max": 50,
        "S_noise": 1.003
    }


    add_spec("edm", "pi", 49, sampler_schedule_path=f"{cfg.base_out}/pi_adaptive/49NFE/data/_t.csv", **edm_churn_params)
    add_spec("edm", "pi", 75, sampler_schedule_path=f"{cfg.base_out}/pi_adaptive/75NFE/data/_t.csv", **edm_churn_params)
    add_spec("edm", "pi", 99, sampler_schedule_path=f"{cfg.base_out}/pi_adaptive/99NFE/data/_t.csv", **edm_churn_params)

    return cfg
