from typing import List, Optional, Tuple, Union, cast
import torch
from torch.linalg import vector_norm
from torch.optim.optimizer import (
    _dispatch_sqrt,
    # _get_capturable_supported_devices,
    _get_value,
    _use_grad_for_differentiable,
    ParamsT,
)


class LAMBAdamW(torch.optim.AdamW):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        LAMB_coefficient: float = 0.001,
    ):
        if not 0 < LAMB_coefficient < 1:
            raise ValueError(f"Invalid LARS_coefficient value: {LAMB_coefficient}")

        self.LAMB_coefficient = LAMB_coefficient

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            foreach=False,
            maximize=maximize,
            capturable=False,
            differentiable=False,
            fused=False,
        )
        super().__init__(params, **defaults)

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: List[torch.Tensor] = []
            grads: List[torch.Tensor] = []
            exp_avgs: List[torch.Tensor] = []
            exp_avg_sqs: List[torch.Tensor] = []
            max_exp_avg_sqs: List[torch.Tensor] = []
            state_steps: List[torch.Tensor] = []
            amsgrad: bool = group["amsgrad"]
            beta1, beta2 = cast(Tuple[float, float], group["betas"])

            _single_tensor_adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                LAMB_coefficient=self.LAMB_coefficient,
            )

        return loss


def _single_tensor_adamw(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    max_exp_avg_sqs: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    grad_scale: Optional[torch.Tensor],
    found_inf: Optional[torch.Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: Union[torch.Tensor, float],
    weight_decay: float,
    eps: float,
    maximize: bool,
    LAMB_coefficient: float,
):
    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            if amsgrad:
                max_exp_avg_sqs[i] = torch.view_as_real(max_exp_avg_sqs[i])
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        step = _get_value(step_t)

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        step_size = lr / bias_correction1

        bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

        # this is not the same as the ratio r_t from the LAMB paper
        # this update variable already includes weight decay
        update = exp_avg.div(denom)

        param_norm = vector_norm(param)
        update_norm = vector_norm(update)

        local_lr = LAMB_coefficient * param_norm.div(update_norm)

        param.addcdiv_(exp_avg, denom, value=-(step_size * local_lr))

        # Lastly, switch back to complex view
        if amsgrad and torch.is_complex(params[i]):
            max_exp_avg_sqs[i] = torch.view_as_complex(max_exp_avg_sqs[i])
