"""
training package — losses, schedulers, and the fine-tuning loop.

All symbols are re-exported here so callers can do:

    from training import fit, MultiTaskLoss, build_optimizer
"""

from __future__ import annotations

try:
    from .losses import (  # type: ignore[import-untyped]
        AsymmetricLoss,
        FocalLoss,
        MultiTaskLoss,
    )
    from .scheduler import (  # type: ignore[import-untyped]
        build_optimizer,
        cosine_schedule_with_warmup,
    )
    from .finetune import (  # type: ignore[import-untyped]
        fit,
        train_one_epoch,
        validate_one_epoch,
    )
    from .pretrain import pretrain_moco  # type: ignore[import-untyped]
except ImportError:
    pass  # torch not installed — training unavailable

__all__ = [
    "FocalLoss",
    "AsymmetricLoss",
    "MultiTaskLoss",
    "cosine_schedule_with_warmup",
    "build_optimizer",
    "train_one_epoch",
    "validate_one_epoch",
    "fit",
]
