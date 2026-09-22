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
    cfg.base_out = "data/image_testing/"

    # Evaluation
    cfg.metrics = [Metrics.FID]
    cfg.feature_path = "refs/img64_features.pkl"
    cfg.stats_path = "refs/img64_stats.pkl"
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

    # GGF specs
    ggf_params = lambda tau_rel, h_0, n: {
        "ode_threshold": 0.05,
        "n_ode_steps": n,
        "tau_a": 0.0078,
        "tau_r": tau_rel,
        "alpha": 0.7,
        "h_start": h_0,
        "max_decrease": 0.2,
        "max_increase": 5,
        "r": 0.1,
        "max_iter": 1000
    }
    add_spec("ggf", "adaptive", 75, **ggf_params(17, 25, 7))

    return cfg
