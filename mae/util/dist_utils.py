
from __future__ import annotations
from typing import Any, List
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

# --------------------- utility base ---------------------

def is_distributed_training_run() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

def get_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

def get_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

def is_main_process() -> bool:
    return get_rank() == 0

def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def _current_device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

# --------------------- seed ---------------------

def set_seed(seed: int = 0) -> None:
    """Allinea la randomizzazione e rende deterministiche le conv di cuDNN."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------- conversioni device ---------------------

def convert_to_distributed_tensor(x: torch.Tensor):
    """Ritorna (tensor_su_device_locale, device_orig) per compat con vecchie API."""
    orig_device = x.device
    target = _current_device()
    if x.device != target:
        x = x.to(target, non_blocking=True)
    return x, orig_device

def convert_to_normal_tensor(x: torch.Tensor, orig_device: torch.device) -> torch.Tensor:
    return x.to(orig_device, non_blocking=True)

# --------------------- gather con grad ---------------------

class GatherLayer(torch.autograd.Function):
    """All-gather che preserva il gradiente in backward."""
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        world_size = get_world_size()
        outputs = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(outputs, x)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grads):
        # somma dei gradienti e ritorno dello slice relativo al proprio rank
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[get_rank()]

def gather_from_all(tensor: torch.Tensor) -> torch.Tensor:
    """
    Concatena il tensore lungo dim=0 da tutti i rank preservando il grad.
    Se non in DDP, ritorna il tensore invariato.
    """
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)

    if not is_distributed_training_run():
        return tensor

    tensor, orig_device = convert_to_distributed_tensor(tensor)
    gathered_tensors = GatherLayer.apply(tensor)
    gathered_tensors = [convert_to_normal_tensor(t, orig_device) for t in gathered_tensors]
    return torch.cat(gathered_tensors, dim=0)

# --------------------- gather eterogeneo (batch diversi) ---------------------

def all_gather_sizes(x: torch.Tensor) -> List[int]:
    """Raccoglie le dimensioni lungo dim=0 da tutti i rank."""
    device = x.device if x.is_cuda else _current_device()
    my = torch.tensor([x.shape[0]], device=device, dtype=torch.int64)
    sizes = [torch.zeros_like(my) for _ in range(get_world_size())]
    dist.all_gather(sizes, my)
    return [int(s.item()) for s in sizes]

def all_gather_heterogeneous(sizes: List[int], x: torch.Tensor) -> List[torch.Tensor]:
    """
    All-gather quando i batch per rank hanno size diverse lungo dim=0.
    Effettua padding al max_len, all_gather e poi rimuove il padding.
    """
    if not is_distributed_training_run():
        return [x]

    max_len = max(sizes)
    if x.size(0) < max_len:
        pad = [0, 0] * x.dim()
        # pad sull'ultima coppia riferita alla prima dimensione (dim=0)
        pad[(x.dim() - 1) * 2] = 0
        pad[(x.dim() - 1) * 2 + 1] = max_len - x.size(0)
        x_padded = F.pad(x, pad)
    else:
        x_padded = x

    world_size = get_world_size()
    out_list = [torch.empty_like(x_padded) for _ in range(world_size)]
    dist.all_gather(out_list, x_padded)

    trimmed = []
    for t, ln in zip(out_list, sizes):
        trimmed.append(t[:ln])
    return trimmed
