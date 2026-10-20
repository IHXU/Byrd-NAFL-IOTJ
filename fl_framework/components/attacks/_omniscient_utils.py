from typing import List, Tuple

import torch


def flatten_gradients(gradients: List[torch.Tensor], device=None) -> torch.Tensor:
    """Flatten one client's gradient tensor list into a single vector."""
    if device is None:
        device = gradients[0].device
    return torch.cat([g.to(device).reshape(-1) for g in gradients])


def stack_honest_gradients(all_honest_gradients: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Size]]:
    """Return an [num_honest, dim] matrix and the original tensor shapes."""
    if not all_honest_gradients:
        raise ValueError("Omniscient attacks require context.all_honest_gradients.")

    ref_grad = all_honest_gradients[0]
    if not ref_grad:
        raise ValueError("Empty honest gradient list.")

    device = ref_grad[0].device
    shapes = [g.shape for g in ref_grad]
    flat = torch.stack([flatten_gradients(client_grad, device) for client_grad in all_honest_gradients])
    return flat, shapes


def unflatten_gradient(flat_gradient: torch.Tensor, shapes: List[torch.Size]) -> List[torch.Tensor]:
    """Convert one flat gradient vector back to the framework's tensor-list format."""
    result = []
    offset = 0
    for shape in shapes:
        numel = 1
        for dim in shape:
            numel *= dim
        result.append(flat_gradient[offset:offset + numel].view(shape).clone())
        offset += numel
    return result


def honest_mean(flat_honest: torch.Tensor) -> torch.Tensor:
    return flat_honest.mean(dim=0)


def max_pairwise_distance(flat_honest: torch.Tensor) -> torch.Tensor:
    if flat_honest.shape[0] <= 1:
        return torch.tensor(0.0, device=flat_honest.device)
    distances = torch.cdist(flat_honest, flat_honest, p=2)
    return distances.max()


def max_distance_sum(flat_honest: torch.Tensor) -> torch.Tensor:
    if flat_honest.shape[0] <= 1:
        return torch.tensor(0.0, device=flat_honest.device)
    distances = torch.cdist(flat_honest, flat_honest, p=2)
    return distances.sum(dim=1).max()


def normalize_direction(direction: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(direction)
    if norm > 0:
        return direction / norm

    fallback_norm = torch.norm(fallback)
    if fallback_norm > 0:
        return fallback / fallback_norm

    out = torch.zeros_like(direction)
    if out.numel() > 0:
        out[0] = 1.0
    return out
