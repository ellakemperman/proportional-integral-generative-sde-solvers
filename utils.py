import torch


def broadcast_vector(vector: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    return vector.view(tensor.shape[0], *([1] * (tensor.dim() - 1)))