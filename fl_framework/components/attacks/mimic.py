from typing import List
import torch
from .base_attack import BaseAttack, Context
from ._omniscient_utils import stack_honest_gradients, unflatten_gradient

class Mimic(BaseAttack):
    """Eq. (22): copy the honest gradient with maximum first-PC projection."""
    @torch.no_grad()
    def attack(self, context: Context) -> List[torch.Tensor]:
        key = ('mimic', context.current_round, context.current_step)
        if key not in context.extra:
            x, shapes = stack_honest_gradients(context.all_honest_gradients)
            centered = x-x.mean(0)
            values, u = torch.linalg.eigh(centered @ centered.T)
            if values[-1] <= torch.finfo(x.dtype).eps:
                target = 0
            else:
                pc = centered.T @ u[:, -1]
                pc /= pc.norm()
                # Principal component signs are arbitrary; fix a convention.
                pc *= torch.where(pc[pc.abs().argmax()] < 0, -1., 1.)
                target = int((x @ pc).argmax())
            context.extra[key] = unflatten_gradient(x[target], shapes)
        return [g.clone() for g in context.extra[key]]
