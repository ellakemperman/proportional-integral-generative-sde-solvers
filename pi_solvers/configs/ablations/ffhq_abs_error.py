from ml_collections import config_dict

from pi_solvers.evaluation.metrics import Metrics


def get_config():
    cfg = config_dict.ConfigDict()

    # Sampling
    cfg.model = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-ve.pkl"
    cfg.n_samples = 50000
    cfg.sampling_batch_size = 128
    cfg.device = "cuda:0"
    cfg.seed = 0
    cfg.ode = False
    cfg.exist_okay = True
    cfg.base_out = "data/ffhq/"

    # Evaluation
    cfg.metrics = [Metrics.FID]
    cfg.feature_path = "refs/ffhq_features.pkl"
    cfg.stats_path = "refs/ffhq_stats.pkl"
    cfg.eval_batch_size = 1024
    cfg.metrics_out = cfg.base_out

    cfg.specs = []

    def add_spec(
            sampler_name: str,
            sampler_schedule_name: str,
            nfe: int,
            ode_threshold: float = 0.05,
            sampler_schedule_path: str | None = None,
            raw_path: bool = False,
            **sampler_kwargs
    ):
        spec = config_dict.ConfigDict()
        spec.sampler_name = sampler_name
        spec.sampler_schedule_name = sampler_schedule_name
        spec.nfe = nfe
        spec.raw_path = raw_path
        if raw_path:
            spec.sampler_schedule_path = sampler_schedule_path
        else:
            spec.sampler_schedule_path = f"{cfg.base_out}/{sampler_schedule_path}"
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
        "max_iter": 1000,
        "abs_error": True,
    }

    add_spec("pi", "adaptive", 49, **pi_params(95, 45, 5))
    add_spec("pi", "adaptive", 75, **pi_params(89, 40, 7))
    add_spec("pi", "adaptive", 99, **pi_params(70, 35, 10))

    return cfg
