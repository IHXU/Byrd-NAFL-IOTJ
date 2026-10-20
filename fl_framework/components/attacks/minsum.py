from typing import List

import torch

from ._omniscient_utils import (
    honest_mean,
    max_distance_sum,
    normalize_direction,
    stack_honest_gradients,
    unflatten_gradient,
)
from .base_attack import BaseAttack, Context


class MinSum(BaseAttack):
    """
    Omniscient MinSum attack.

    It maximizes adversarial displacement while keeping the sum of distances to
    honest gradients within the largest honest-to-honest distance sum.
    """

    def __init__(
        self,
        direction: str = "negative_mean",
        max_iter: int = 30,
        initial_gamma: float = 1.0,
        eps: float = 1e-12,
    ):
        self.direction = direction
        self.max_iter = int(max_iter)
        self.initial_gamma = float(initial_gamma)
        self.eps = float(eps)
        self.current_round = None
        self.current_step = None
        self.malicious_grad = None

    @torch.no_grad()
    def attack(self, context: Context) -> List[torch.Tensor]:
        cache_key = (
            "attack",
            "minsum",
            context.current_round,
            context.current_step,
            self.direction,
            self.max_iter,
            self.initial_gamma,
        )
        if context.extra is not None and cache_key in context.extra:
            return [g.clone() for g in context.extra[cache_key]]
        if (
            self.current_round == context.current_round
            and self.current_step == context.current_step
            and self.malicious_grad is not None
        ):
            return [g.clone() for g in self.malicious_grad]

        flat_honest, shapes = stack_honest_gradients(context.all_honest_gradients)
        mean = honest_mean(flat_honest)
        direction = self._direction(flat_honest, mean)
        limit = max_distance_sum(flat_honest)

        gamma = self._binary_search_gamma(
            mean=mean,
            direction=direction,
            flat_honest=flat_honest,
            limit=limit,
        )
        malicious = mean + gamma * direction

        self.current_round = context.current_round
        self.current_step = context.current_step
        self.malicious_grad = unflatten_gradient(malicious, shapes)
        if context.extra is not None:
            context.extra[cache_key] = [g.clone() for g in self.malicious_grad]
        return [g.clone() for g in self.malicious_grad]

    def _direction(self, flat_honest: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        if self.direction == "sign":
            raw = -torch.sign(mean)
        elif self.direction == "std":
            raw = -flat_honest.std(dim=0, unbiased=False)
        else:
            raw = -mean
        return normalize_direction(raw, -mean)

    def _binary_search_gamma(
        self,
        mean: torch.Tensor,
        direction: torch.Tensor,
        flat_honest: torch.Tensor,
        limit: torch.Tensor,
    ) -> torch.Tensor:
        if limit <= self.eps:
            return torch.tensor(0.0, device=flat_honest.device)

        low = torch.tensor(0.0, device=flat_honest.device)
        high = torch.tensor(self.initial_gamma, device=flat_honest.device)

        def feasible(gamma: torch.Tensor) -> bool:
            candidate = mean + gamma * direction
            dist_sum = torch.norm(flat_honest - candidate.unsqueeze(0), dim=1).sum()
            return bool(dist_sum <= limit + self.eps)

        while feasible(high):
            low = high
            high = high * 2.0
            if high > limit * 4.0 + 1.0:
                break

        for _ in range(self.max_iter):
            mid = (low + high) / 2.0
            if feasible(mid):
                low = mid
            else:
                high = mid
        return low
