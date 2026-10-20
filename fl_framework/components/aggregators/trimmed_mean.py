from typing import Any, Dict, List, Optional

import torch

from .base_aggregator import BaseAggregator


class TrimmedMeanAggregator(BaseAggregator):
    """
    Coordinate-wise trimmed mean aggregation.

    For each gradient coordinate, the largest and smallest values are removed
    before averaging the remaining values.
    """

    def __init__(
        self,
        trim_ratio: Optional[float] = None,
        num_trim: Optional[int] = None,
    ):
        if trim_ratio is None and num_trim is None:
            trim_ratio = 0.2
        if trim_ratio is not None and not 0.0 <= float(trim_ratio) < 0.5:
            raise ValueError("trim_ratio must be in [0.0, 0.5).")
        if num_trim is not None and int(num_trim) < 0:
            raise ValueError("num_trim cannot be negative.")

        self.trim_ratio = None if trim_ratio is None else float(trim_ratio)
        self.num_trim = None if num_trim is None else int(num_trim)

    @torch.no_grad()
    def aggregate(self, all_gradients: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        if not all_gradients:
            return []

        ref_grad = all_gradients[0]
        if not ref_grad:
            return []

        num_clients = len(all_gradients)
        trim = self._trim_count(num_clients)
        if 2 * trim >= num_clients:
            raise ValueError(
                f"Cannot trim {trim} values from each side with {num_clients} clients."
            )

        result = []
        device = ref_grad[0].device
        for layer_idx, ref_layer in enumerate(ref_grad):
            layer_values = torch.stack([
                client_grad[layer_idx].to(device).reshape(-1)
                for client_grad in all_gradients
            ])
            sorted_values, _ = torch.sort(layer_values, dim=0)
            kept_values = sorted_values[trim:num_clients - trim]
            result.append(kept_values.mean(dim=0).view(ref_layer.shape))

        return result

    def _trim_count(self, num_clients: int) -> int:
        if self.num_trim is not None:
            return self.num_trim
        return int(num_clients * self.trim_ratio)

    def get_state(self) -> Dict[str, Any]:
        return None

    def set_state(self, state: Dict[str, Any]):
        pass
