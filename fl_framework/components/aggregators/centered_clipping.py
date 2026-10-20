from typing import Any, Dict, List, Optional

import torch

from .base_aggregator import BaseAggregator


class CenteredClippingAggregator(BaseAggregator):
    """
    Centered clipping aggregation.

    Each client update is clipped around a running center, then the clipped
    updates are averaged. The center is updated to the returned aggregate and
    stored in the aggregator state for checkpoint/resume.
    """

    def __init__(
        self,
        tau: float = None,
        n_iter: int = 1,
        center: str = "previous",
        initial_center: Optional[str] = None,
        tau_quantile: float = 0.5,
        tau_multiplier: float = 1.0,
        eps: float = 1e-12,
    ):
        self.tau = None if tau is None else float(tau)
        self.n_iter = int(n_iter)
        self.center_mode = center
        if initial_center not in (None, "mean", "zero", "coordinate_median"):
            raise ValueError(
                "initial_center must be one of: mean, zero, coordinate_median."
            )
        self.initial_center = initial_center
        self.tau_quantile = float(tau_quantile)
        self.tau_multiplier = float(tau_multiplier)
        self.eps = float(eps)
        self.center = None

    @torch.no_grad()
    def aggregate(self, all_gradients: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        if not all_gradients:
            return []
        ref_grad = all_gradients[0]
        if not ref_grad:
            return []

        flat_grads, shapes = self._flatten_all(all_gradients)

        if self.center is None or self.center.numel() != flat_grads.shape[1]:
            center = self._initial_center(flat_grads)
        else:
            center = self.center.to(flat_grads.device)

        for _ in range(max(1, self.n_iter)):
            diff = flat_grads - center.unsqueeze(0)
            norms = torch.norm(diff, dim=1).clamp_min(self.eps)
            tau = self._current_tau(norms)
            scales = torch.clamp(tau / norms, max=1.0)
            clipped = center.unsqueeze(0) + diff * scales.unsqueeze(1)
            center = clipped.mean(dim=0)

        self.center = center.detach().clone()
        return self._unflatten(center, shapes)

    def _flatten_all(self, all_gradients: List[List[torch.Tensor]]):
        device = all_gradients[0][0].device
        shapes = [g.shape for g in all_gradients[0]]
        flat = torch.stack([
            torch.cat([g.to(device).reshape(-1) for g in client_grad])
            for client_grad in all_gradients
        ])
        return flat, shapes

    def _unflatten(self, flat_gradient: torch.Tensor, shapes: List[torch.Size]) -> List[torch.Tensor]:
        result = []
        offset = 0
        for shape in shapes:
            numel = 1
            for dim in shape:
                numel *= dim
            result.append(flat_gradient[offset:offset + numel].view(shape).clone())
            offset += numel
        return result

    def get_state(self) -> Dict[str, Any]:
        return {
            "center": self.center.detach().cpu() if self.center is not None else None,
            "tau": self.tau,
            "n_iter": self.n_iter,
            "center_mode": self.center_mode,
            "initial_center": self.initial_center,
            "tau_quantile": self.tau_quantile,
            "tau_multiplier": self.tau_multiplier,
        }

    def set_state(self, state: Dict[str, Any]):
        if not state:
            return
        self.center = state.get("center")
        tau = state.get("tau", self.tau)
        self.tau = None if tau is None else float(tau)
        self.n_iter = int(state.get("n_iter", self.n_iter))
        self.center_mode = state.get("center_mode", self.center_mode)
        self.initial_center = state.get("initial_center", self.initial_center)
        self.tau_quantile = float(state.get("tau_quantile", self.tau_quantile))
        self.tau_multiplier = float(state.get("tau_multiplier", self.tau_multiplier))

    def _current_tau(self, norms: torch.Tensor) -> torch.Tensor:
        if self.tau is not None:
            return torch.tensor(self.tau, device=norms.device, dtype=norms.dtype)
        q = min(max(self.tau_quantile, 0.0), 1.0)
        return torch.quantile(norms.detach(), q) * self.tau_multiplier

    def _initial_center(self, flat_grads: torch.Tensor) -> torch.Tensor:
        mode = self.initial_center
        if mode is None:
            # Preserve the behavior of existing configurations that use
            # center="zero"; all other legacy configurations started at mean.
            mode = "zero" if self.center_mode == "zero" else "mean"
        if mode == "zero":
            return torch.zeros_like(flat_grads[0])
        if mode == "coordinate_median":
            return flat_grads.median(dim=0).values
        return flat_grads.mean(dim=0)
