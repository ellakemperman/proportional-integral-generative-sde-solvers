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

    # EM
    ret_path = "refs/ret_ffhq.pt"

    add_spec("euler-maruyama", "entropic", 49, sampler_schedule_path=ret_path)
    add_spec("euler-maruyama", "entropic", 75, sampler_schedule_path=ret_path)
    add_spec("euler-maruyama", "entropic", 99, sampler_schedule_path=ret_path)

    # Stochastic Heun
    add_spec("heun", "entropic", 49, sampler_schedule_path=ret_path)
    add_spec("heun", "entropic", 75, sampler_schedule_path=ret_path)
    add_spec("heun", "entropic", 99, sampler_schedule_path=ret_path)

    # EDM
    edm_churn_params = {
        "S_churn": 40,
        "S_min": 0.05,
        "S_max": 50,
        "S_noise": 1.003
    }

    add_spec("edm", "entropic", 49, **edm_churn_params, sampler_schedule_path=ret_path)
    add_spec("edm", "entropic", 75, **edm_churn_params, sampler_schedule_path=ret_path)
    add_spec("edm", "entropic", 99, **edm_churn_params, sampler_schedule_path=ret_path)

    return cfg
