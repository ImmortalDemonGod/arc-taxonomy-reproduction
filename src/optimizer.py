from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    From-scratch implementation of AdamW (Decoupled Weight Decay Regularization).

    References:
    - Loshchilov & Hutter, 2019: Decoupled Weight Decay Regularization.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (alpha).
        betas: Coefficients used for computing running averages of gradient and its square.
        eps: Term added to the denominator for numerical stability.
        weight_decay: Weight decay coefficient (decoupled).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        b1, b2 = betas
        if not 0.0 <= b1 < 1.0 or not 0.0 <= b2 < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss (optional).
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                # Decoupled weight decay
                if wd != 0.0:
                    p.data.add_(p.data, alpha=-lr * wd)

                # Adam moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t
                exp_avg_hat = exp_avg / bias_correction1
                exp_avg_sq_hat = exp_avg_sq / bias_correction2

                denom = exp_avg_sq_hat.sqrt().add_(eps)

                # Parameter update
                p.data.addcdiv_(exp_avg_hat, denom, value=-lr)

        return loss
