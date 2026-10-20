"""Server-side Nesterov momentum for Byrd-NAFL."""
from typing import List
import torch
from .base_optimizer import BaseOptimizer
from fl_framework.core.hooks import HookType, Context, hook_registry

class SGDNesterov(BaseOptimizer):
    """z_next = beta*z + g; y = beta*z_next + g; x_next = x - lr*y."""
    def __init__(self, lr: float, weight_decay: float = 0.0, momentum: float = 0.0):
        super().__init__(lr=lr)
        self.weight_decay = weight_decay
        self.momentum = momentum
        self._z: List[torch.Tensor] | None = None
        self._y: List[torch.Tensor] | None = None

    def register_hooks(self):
        super().register_hooks()
        hook_registry.register(HookType.AFTER_COMPUTE, self._weight_decay)

    @torch.no_grad()
    def _weight_decay(self, context: Context):
        client_id = context.current_client_id
        client = context.clients[client_id]
        if client.client_type == 'Byzantine' and not client.if_compute_grad:
            return
        torch._foreach_add_(context.grad[client_id], list(client.model.parameters()), alpha=self.weight_decay)

    @torch.no_grad()
    def step(self, server, aggregated_grad):
        if server is None or aggregated_grad is None:
            return
        model = server.model
        if self._z is None:
            self._z = [torch.zeros_like(p) for p in model.parameters()]
            self._y = [torch.zeros_like(p) for p in model.parameters()]
        self._z = [z.to(p.device) for z,p in zip(self._z,model.parameters())]
        self._y = [y.to(p.device) for y,p in zip(self._y,model.parameters())]
        for i,(param,grad) in enumerate(zip(model.parameters(),aggregated_grad)):
            g = grad.to(param.device)
            self._z[i].mul_(self.momentum).add_(g)
            self._y[i].copy_(self._z[i]).mul_(self.momentum).add_(g)
            param.add_(self._y[i],alpha=-self.lr)

    def get_state(self):
        state = {'lr':self.lr, 'momentum':self.momentum, 'weight_decay':self.weight_decay}
        if self._z is not None:
            state['z'] = [z.cpu() for z in self._z]
            state['y'] = [y.cpu() for y in self._y]
        return state

    def set_state(self,state):
        self.lr = state.get('lr',self.lr)
        self.momentum = state.get('momentum',self.momentum)
        self.weight_decay = state.get('weight_decay',self.weight_decay)
        if state.get('z') is not None and state.get('y') is not None:
            self._z = [z.clone() for z in state['z']]
            self._y = [y.clone() for y in state['y']]
