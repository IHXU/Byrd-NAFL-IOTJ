from typing import List
import torch
from .base_attack import BaseAttack, Context

class SignFlipping(BaseAttack):
    """Omniscient sign flipping, g* = -mean(g_h), as in Eq. (21)."""
    @torch.no_grad()
    def attack(self, context: Context) -> List[torch.Tensor]:
        honest = context.all_honest_gradients
        if not honest:
            raise ValueError('Sign flipping requires current honest gradients.')
        return [-torch.stack([g[i].to(ref.device) for g in honest]).mean(0)
                for i, ref in enumerate(honest[0])]
