from typing import List
from statistics import NormalDist

import torch

from .base_attack import BaseAttack, Context


class ALIE(BaseAttack):
    """
    A Little Is Enough (ALIE) omniscient attack.

    The malicious update is mean + direction * z * std computed coordinate-wise
    from honest gradients. The usual ALIE direction is -1.
    """

    def __init__(self, z=None, direction: float = -1.0, unbiased_std: bool = False):
        self.z = z
        self.direction = float(direction)
        self.unbiased_std = bool(unbiased_std)
        self.current_round = None
        self.current_step = None
        self.malicious_grad = None

    @torch.no_grad()
    def attack(self, context: Context) -> List[torch.Tensor]:
        """
        Generates malicious gradients to counteract honest gradients.
        """
        cache_key = (
            "attack",
            "alie",
            context.current_round,
            context.current_step,
            self.z,
            self.direction,
            self.unbiased_std,
            context.num_byzantine,
        )
        if context.extra is not None and cache_key in context.extra:
            return [g.clone() for g in context.extra[cache_key]]
        if (
            self.current_round == context.current_round
            and self.current_step == context.current_step
            and self.malicious_grad is not None
        ):
            return [g.clone() for g in self.malicious_grad]

        if self.z is None:
            num_honest = len(context.all_honest_gradients)
            num_byz = context.num_byzantine
            if num_byz is None:
                num_byz = len(context.clients) - num_honest
            s = int((num_honest + num_byz) / 2 + 1) - num_byz
            self.z = float(NormalDist().inv_cdf((num_honest + num_byz - s) / (num_byz + num_honest)))

        ref_grad = context.all_honest_gradients[0]
        device = ref_grad[0].device
        honest_grad = [
            torch.stack([client_grad[i].to(device) for client_grad in context.all_honest_gradients])
            for i in range(len(ref_grad))
        ]
        mean_grad = [torch.mean(layer_gradients, dim=0) for layer_gradients in honest_grad]
        std_grad = [
            torch.std(layer_gradients, dim=0, unbiased=self.unbiased_std)
            for layer_gradients in honest_grad
        ]

        malicious_grad = [
            mean + self.direction * self.z * std
            for mean, std in zip(mean_grad, std_grad)
        ]
        self.current_round = context.current_round
        self.current_step = context.current_step
        self.malicious_grad = [
            layer_grad.reshape(ref_grad[i].shape).clone()
            for i, layer_grad in enumerate(malicious_grad)
        ]
        if context.extra is not None:
            context.extra[cache_key] = [g.clone() for g in self.malicious_grad]
        return [g.clone() for g in self.malicious_grad]
