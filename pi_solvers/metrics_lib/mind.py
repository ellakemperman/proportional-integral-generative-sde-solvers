import torch


def monge_inception_distance_torch(
        x_hat: torch.Tensor,
        x: torch.Tensor,
        seed: int,
        n_projections: int = 100
):
    """
    Computes the MIND metric from the paper MIND: Monge Inception Distance for Generative Models
    Evaluation: https://arxiv.org/html/2605.06797v1#A3.F10.sf2

    :param x_hat: The generated features
    :param x: Ground truth
    :param seed: Random seed
    :param n_projections: Number of projections to use
    :return: The value of the MIND metric
    """

    n_samples, d = x_hat.shape
    assert n_samples <= x.shape[0], "Ground truth needs to have at least as many samples as predicted"

    x = x[:n_samples]

    alpha = 3 * d
    generator = torch.Generator(device=x.device).manual_seed(seed)

    u_proj = torch.randn(
        (n_projections, d),
        generator=generator,
        dtype=x.dtype,
        device=x.device
    )

    u_proj /= torch.linalg.norm(u_proj, dim=-1, keepdim=True)

    x_proj = u_proj @ x.T
    x_hat_proj = x_hat @ x_hat.T

    dists = torch.mean(
        (
            torch.topk(x_hat_proj, n_samples, dim=-1).values -
            torch.topk(x_proj, n_samples, dim=-1).values
        )**2,
        dim=-1
    )

    return alpha * torch.mean(dists)