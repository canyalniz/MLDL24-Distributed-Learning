from typing import List, Optional
import torch
from torch import Tensor
from torch.optim.optimizer import _use_grad_for_differentiable
from torch.linalg import vector_norm


class LARSSGD(torch.optim.SGD):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov=False,
        *,
        maximize: bool = False,
        differentiable: bool = False,
        LARS_coefficient=0.001,
    ):
        if not 0 < LARS_coefficient < 1:
            raise ValueError(f"Invalid LARS_coefficient value: {LARS_coefficient}")

        self.LARS_coefficient = LARS_coefficient

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            foreach=False,
            differentiable=differentiable,
            fused=False,
        )
        super().__init__(params, **defaults)

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            momentum_buffer_list: List[Optional[Tensor]] = []

            has_sparse_grad = self._init_group(
                group, params, grads, momentum_buffer_list
            )

            _single_tensor_larssgd(
                params,
                grads,
                momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                maximize=group["maximize"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                LARS_coefficient=self.LARS_coefficient,
            )

            if group["momentum"] != 0:
                # update momentum_buffers in state
                for p, momentum_buffer in zip(params, momentum_buffer_list):
                    state = self.state[p]
                    state["momentum_buffer"] = momentum_buffer

        return loss

    # LARSSGD.__doc__ = r"""Implements stochastic gradient descent with LARS."""


def _single_tensor_larssgd(
    params: List[Tensor],
    d_p_list: List[torch.Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    LARS_coefficient: float,
):
    assert grad_scale is None and found_inf is None

    for i, param in enumerate(params):
        d_p = d_p_list[i] if not maximize else -d_p_list[i]

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        layer_norm = vector_norm(param)
        update_norm = vector_norm(d_p)

        local_lr = LARS_coefficient * layer_norm / update_norm

        param.add_(d_p, alpha=-(lr * local_lr))
