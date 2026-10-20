from typing import Any, Dict, List, Optional

import torch

from .base_aggregator import BaseAggregator


class BulyanAggregator(BaseAggregator):
    """
    Bulyan aggregation with Multi-Krum selection and coordinate-wise filtering.

    The strict theoretical setting requires enough clients relative to the
    Byzantine count. With strict=False, the second-stage trim count is reduced
    when needed so high-Byzantine-ratio experiments can still run.
    """

    def __init__(
        self,
        num_byzantine: int,
        selection_size: Optional[int] = None,
        trim_ratio: Optional[float] = None,
        strict: bool = False,
    ):
        if int(num_byzantine) < 0:
            raise ValueError("num_byzantine cannot be negative.")
        if selection_size is not None and int(selection_size) <= 0:
            raise ValueError("selection_size must be positive.")
        if trim_ratio is not None and not 0.0 <= float(trim_ratio) < 0.5:
            raise ValueError("trim_ratio must be in [0.0, 0.5).")

        self.num_byzantine = int(num_byzantine)
        self.selection_size = None if selection_size is None else int(selection_size)
        self.trim_ratio = None if trim_ratio is None else float(trim_ratio)
        self.strict = bool(strict)

    @torch.no_grad()
    def aggregate(self, all_gradients: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        if not all_gradients:
            return []

        ref_grad = all_gradients[0]
        if not ref_grad:
            return []

        num_clients = len(all_gradients)
        if num_clients <= 2 * self.num_byzantine + 2:
            raise ValueError(
                f"Bulyan requires more clients than 2f + 2; got n={num_clients}, "
                f"f={self.num_byzantine}."
            )
        if self.strict and num_clients <= 4 * self.num_byzantine + 3:
            raise ValueError(
                f"Strict Bulyan requires n > 4f + 3; got n={num_clients}, "
                f"f={self.num_byzantine}."
            )

        device = ref_grad[0].device
        flat_gradients = torch.stack([
            torch.cat([g.to(device).reshape(-1) for g in client_grad])
            for client_grad in all_gradients
        ])

        selected_indices = self._multi_krum_select(flat_gradients, num_clients)
        selected = flat_gradients[selected_indices]

        trim = self._trim_count(selected.shape[0])
        if 2 * trim >= selected.shape[0]:
            if self.strict:
                raise ValueError(
                    f"Cannot trim f={trim} from selected size {selected.shape[0]}."
                )
            trim = max((selected.shape[0] - 1) // 2, 0)

        aggregate_flat = self._coordinate_filter(selected, trim)
        return self._unflatten(aggregate_flat, [g.shape for g in ref_grad])

    def _multi_krum_select(
        self,
        flat_gradients: torch.Tensor,
        num_clients: int,
    ) -> torch.Tensor:
        theta = self.selection_size
        if theta is None:
            theta = num_clients - 2 * self.num_byzantine
        theta = min(max(theta, 1), num_clients)

        distances = torch.cdist(flat_gradients, flat_gradients, p=2) ** 2
        neighbor_count = min(
            max(num_clients - self.num_byzantine - 1, 1),
            num_clients,
        )
        top_dists, _ = torch.topk(
            distances,
            neighbor_count,
            dim=1,
            largest=False,
        )
        scores = top_dists.sum(dim=1)
        _, selected_indices = torch.topk(scores, theta, largest=False)
        return selected_indices

    def _trim_count(self, selected_size: int) -> int:
        if self.trim_ratio is not None:
            return int(selected_size * self.trim_ratio)
        return self.num_byzantine

    def _coordinate_filter(self, selected: torch.Tensor, trim: int) -> torch.Tensor:
        if trim == 0:
            return selected.mean(dim=0)

        median = selected.median(dim=0).values
        distances = torch.abs(selected - median.unsqueeze(0))
        kept_count = selected.shape[0] - 2 * trim
        kept_count = max(kept_count, 1)
        _, kept_indices = torch.topk(
            distances,
            kept_count,
            dim=0,
            largest=False,
        )
        kept_values = torch.gather(selected, 0, kept_indices)
        return kept_values.mean(dim=0)

    def _unflatten(
        self,
        flat_gradient: torch.Tensor,
        shapes: List[torch.Size],
    ) -> List[torch.Tensor]:
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
        return None

    def set_state(self, state: Dict[str, Any]):
        pass
