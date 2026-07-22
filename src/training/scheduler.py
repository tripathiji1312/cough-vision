"""
LR scheduler: cosine annealing with linear warm-up.

Used for both self-supervised pretraining (MoCo/DINO) and supervised
fine-tuning of the CNN+ViT ensemble.
"""

from __future__ import annotations

import math
from typing import Any

_TORCH_AVAILABLE = False
try:
    import torch as _t  # type: ignore[import-untyped]
    _TORCH_AVAILABLE = True
except ImportError:
    _t = None  # type: ignore[assignment]


def _require(flag: bool, pkg: str) -> None:
    if not flag:
        raise ImportError(f"{pkg} is required. pip install {pkg}")


def cosine_schedule_with_warmup(
    optimizer: Any,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_fraction: float = 0.01,
) -> Any:
    """
    Cosine LR schedule with linear warm-up.

    LR ramps linearly from 0 → base_lr over ``num_warmup_steps``,
    then follows a cosine decay down to ``base_lr * min_lr_fraction``.

    Args:
        optimizer:           PyTorch optimizer.
        num_warmup_steps:    Number of warm-up steps (epochs × steps/epoch).
        num_training_steps:  Total training steps.
        min_lr_fraction:     Floor as a fraction of the peak LR.

    Returns:
        A ``torch.optim.lr_scheduler.LambdaLR`` scheduler.
    """
    _require(_TORCH_AVAILABLE, "torch")
    import torch.optim.lr_scheduler as sched  # type: ignore[import-untyped]

    def lr_lambda(current_step: int) -> float:
        try:
            if current_step < num_warmup_steps:
                return float(current_step) / max(1, num_warmup_steps)
            progress = float(current_step - num_warmup_steps) / max(
                1, num_training_steps - num_warmup_steps
            )
            cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr_fraction, cosine_val)
        except (ZeroDivisionError, ValueError):
            return min_lr_fraction

    return sched.LambdaLR(optimizer, lr_lambda)


def build_optimizer(
    model: Any,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Any:
    """
    AdamW optimizer with discriminative learning rates.

    Backbone parameters use backbone_lr (preserves pretrained representations).
    Head/fusion parameters use head_lr.

    Args:
        model:        TBEnsemble nn.Module (must have .get_parameter_groups()).
        backbone_lr:  LR for frozen-unfrozen backbone (1e-5 per plan).
        head_lr:      LR for classification/fusion heads (1e-3 per plan).
        weight_decay: L2 regularisation (1e-4 per plan).

    Returns:
        torch.optim.AdamW optimizer.
    """
    _require(_TORCH_AVAILABLE, "torch")
    import torch.optim as optim  # type: ignore[import-untyped]

    try:
        param_groups = model.get_parameter_groups(backbone_lr, head_lr)
    except AttributeError:
        # Fallback: treat all parameters uniformly
        param_groups = [{"params": list(model.parameters()), "lr": head_lr}]

    try:
        return optim.AdamW(param_groups, weight_decay=weight_decay)
    except Exception as exc:
        raise RuntimeError(f"Failed to build AdamW optimizer: {exc}") from exc
