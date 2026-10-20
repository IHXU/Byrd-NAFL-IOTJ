from typing import List

import torch

from ._omniscient_utils import honest_mean, stack_honest_gradients, unflatten_gradient
from .base_attack import BaseAttack, Context


class IPM(BaseAttack):
    """
    Inner Product Manipulation attack.

    Sends a scaled negative honest mean. With auto_tune=True, alpha is reduced
    by binary search until the malicious vector passes a Krum-style score test.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        auto_tune: bool = True,
        num_byzantine: int = None,
        max_iter: int = 30,
        eps: float = 1e-12,
    ):
        self.alpha = float(alpha)
        self.auto_tune = bool(auto_tune)
        self.num_byzantine = num_byzantine
        self.max_iter = int(max_iter)
        self.eps = float(eps)
        self.current_round = None
        self.current_step = None
        self.malicious_grad = None

    @torch.no_grad()
    def attack(self, context: Context) -> List[torch.Tensor]:
        cache_key = (
            "attack",
            "ipm",
            context.current_round,
            context.current_step,
            self.alpha,
            self.auto_tune,
            self.num_byzantine,
            context.num_byzantine,
            self.max_iter,
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

        alpha = self.alpha
        if self.auto_tune:
            alpha = self._tune_alpha_for_krum(mean, flat_honest, context)

        malicious = -alpha * mean
        self.current_round = context.current_round
        self.current_step = context.current_step
        self.malicious_grad = unflatten_gradient(malicious, shapes)
        if context.extra is not None:
            context.extra[cache_key] = [g.clone() for g in self.malicious_grad]
        return [g.clone() for g in self.malicious_grad]

    def _tune_alpha_for_krum(self, mean: torch.Tensor, flat_honest: torch.Tensor, context: Context) -> float:
        num_honest = flat_honest.shape[0]
        num_byz = context.num_byzantine
        if num_byz is None:
            num_byz = self.num_byzantine
        if num_byz is None:
            num_byz = max(0, len(context.clients) - num_honest)
        if num_byz <= 0:
            return self.alpha

        total_clients = num_honest + num_byz
        k = total_clients - num_byz - 1
        if k <= 0:
            return self.alpha

        def krum_scores(points: torch.Tensor) -> torch.Tensor:
            distances = torch.cdist(points, points, p=2) ** 2
            top_k, _ = torch.topk(distances, k, dim=1, largest=False)
            return top_k.sum(dim=1)

        def feasible(alpha: float) -> bool:
            malicious = (-alpha * mean).unsqueeze(0).repeat(num_byz, 1)
            points = torch.cat([flat_honest, malicious], dim=0)
            scores = krum_scores(points)
            best = int(torch.argmin(scores).item())
            return best >= num_honest

        if feasible(self.alpha):
            return self.alpha

        low = 0.0
        high = self.alpha
        for _ in range(self.max_iter):
            mid = (low + high) / 2.0
            if feasible(mid):
                low = mid
            else:
                high = mid
        return low
