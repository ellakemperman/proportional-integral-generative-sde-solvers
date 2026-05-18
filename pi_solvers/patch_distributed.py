import torch
import torch.distributed as dist

def _init_noop(**kwargs):
    pass

def _get_rank():
    return 0

def _get_world_size():
    return 1

def _is_initialized():
    return True

def _barrier():
    pass

def _all_reduce(tensor, *args, **kwargs):
    return tensor

def _broadcast(tensor, *args, **kwargs):
    return tensor

dist.init_process_group = lambda *args, **kwargs: None
dist.get_rank = _get_rank
dist.get_world_size = _get_world_size
dist.is_initialized = _is_initialized
dist.barrier = _barrier
dist.all_reduce = _all_reduce
dist.broadcast = _broadcast