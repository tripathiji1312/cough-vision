"""
CNN backbone branch — DenseNet-121 (CheXNet) or EfficientNet-B4.

DenseNet-121 remains the most externally validated backbone for thoracic
disease classification and is the architecture used in the WHO-eligible
CAD4TB lineage.

EfficientNet-B4 achieves AUC 0.95–1.0 on TB-specific manifestations with
the strongest Grad-CAM localization across geographically diverse test sets.

Both models can be initialized from:
  1. ``'imagenet'``      — standard ImageNet weights (timm)
  2. ``'mocov3_cxr'``   — MoCo-v3 CXR self-supervised pretraining
  3. ``'checkpoint'``   — arbitrary local checkpoint

The backbone returns a (B, feature_dim) feature vector from Global Average
Pooling, ready for downstream fusion or classification heads.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

_TORCH_AVAILABLE = False
try:
    import torch as _t  # type: ignore[import-untyped]
    import torch.nn as _nn  # type: ignore[import-untyped]
    _TORCH_AVAILABLE = True
except ImportError:
    _t = None   # type: ignore[assignment]
    _nn = None  # type: ignore[assignment]

_TIMM_AVAILABLE = False
try:
    import timm as _timm  # type: ignore[import-untyped]
    _TIMM_AVAILABLE = True
except ImportError:
    _timm = None  # type: ignore[assignment]


def _require(flag: bool, pkg: str) -> None:
    if not flag:
        raise ImportError(f"{pkg} is required. pip install {pkg}")


# ---------------------------------------------------------------------------
# Supported backbones
# ---------------------------------------------------------------------------

_CNN_TIMM_NAMES: dict[str, str] = {
    "densenet121":   "densenet121",
    "efficientnet_b4": "efficientnet_b4",
    "efficientnet_b0": "efficientnet_b0",   # edge/distilled model
    "efficientnet_b2": "efficientnet_b2",
    "resnet50":      "resnet50",             # baseline only
}

_CNN_FEATURE_DIMS: dict[str, int] = {
    "densenet121":     1024,
    "efficientnet_b4": 1792,
    "efficientnet_b0": 1280,
    "efficientnet_b2": 1408,
    "resnet50":        2048,
}


# ---------------------------------------------------------------------------
# CNN backbone builder
# ---------------------------------------------------------------------------

def build_cnn_backbone(
    name: str = "densenet121",
    pretrained: str = "imagenet",
    pretrained_ckpt: str | None = None,
    drop_rate: float = 0.2,
    in_channels: int = 3,
) -> Any:
    """
    Build a CNN feature extractor (no classification head).

    Args:
        name:            Backbone key from ``_CNN_TIMM_NAMES``.
        pretrained:      ``'imagenet'``, ``'mocov3_cxr'``, or ``'checkpoint'``.
        pretrained_ckpt: Path to a .pt/.pth checkpoint when pretrained='checkpoint'
                         or 'mocov3_cxr'.
        drop_rate:       Dropout rate applied before the GAP output.
        in_channels:     Input channels (3 for RGB-converted greyscale CXR).

    Returns:
        An ``nn.Module`` whose ``.forward(x)`` accepts (B, C, H, W) and
        returns (B, feature_dim).  The feature_dim is accessible as
        ``model.feature_dim``.
    """
    _require(_TORCH_AVAILABLE, "torch")
    _require(_TIMM_AVAILABLE, "timm")
    import torch.nn as nn  # type: ignore[import-untyped]
    import timm  # type: ignore[import-untyped]

    if name not in _CNN_TIMM_NAMES:
        raise ValueError(f"Unknown backbone '{name}'. Choose from: {list(_CNN_TIMM_NAMES)}")

    timm_name = _CNN_TIMM_NAMES[name]
    feature_dim = _CNN_FEATURE_DIMS[name]
    use_imagenet = pretrained == "imagenet"

    class _CNNBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Build timm model — num_classes=0 removes the head, returns GAP features
            self.encoder = timm.create_model(
                timm_name,
                pretrained=use_imagenet,
                num_classes=0,          # remove head → raw GAP features
                global_pool="avg",
                in_chans=in_channels,
                drop_rate=drop_rate,
            )
            self.feature_dim: int = feature_dim
            self.dropout = nn.Dropout(p=drop_rate)

        def forward(self, x: Any) -> Any:
            features = self.encoder(x)   # (B, feature_dim)
            return self.dropout(features)

    model = _CNNBackbone()

    # Load non-ImageNet weights
    if pretrained in ("mocov3_cxr", "checkpoint"):
        if pretrained_ckpt is None:
            warnings.warn(
                f"pretrained='{pretrained}' requested but pretrained_ckpt is None. "
                "The model will use random (non-ImageNet) weights.",
                stacklevel=2,
            )
        else:
            _load_encoder_weights(model.encoder, pretrained_ckpt)

    return model


# ---------------------------------------------------------------------------
# Weight loading helper
# ---------------------------------------------------------------------------

def _load_encoder_weights(encoder: Any, ckpt_path: str) -> None:
    """
    Load encoder weights from a MoCo-v3 or generic checkpoint.

    Handles key-prefix mismatches that arise from MoCo's
    ``module.base_encoder.*`` key structure.
    """
    _require(_TORCH_AVAILABLE, "torch")
    import torch  # type: ignore[import-untyped]

    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        ckpt = torch.load(str(path), map_location="cpu")
    except Exception as exc:
        raise OSError(f"Failed to load checkpoint {path}: {exc}") from exc

    # MoCo-v3 stores weights under 'state_dict' or 'model'
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))

    # Strip common key prefixes
    cleaned: dict[str, Any] = {}
    for k, v in state.items():
        for prefix in ("module.base_encoder.", "base_encoder.", "encoder.", "module."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        cleaned[k] = v

    missing, unexpected = encoder.load_state_dict(cleaned, strict=False)
    if missing:
        warnings.warn(
            f"Missing keys when loading encoder weights ({len(missing)}): "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}",
            stacklevel=3,
        )
    if unexpected:
        warnings.warn(
            f"Unexpected keys ({len(unexpected)}): "
            f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers for staged fine-tuning
# ---------------------------------------------------------------------------

def freeze_backbone(model: Any) -> None:
    """Freeze all encoder parameters (stage 2a: heads-only warm-up)."""
    _require(_TORCH_AVAILABLE, "torch")
    for param in model.encoder.parameters():
        param.requires_grad = False


def unfreeze_backbone(model: Any) -> None:
    """Unfreeze all encoder parameters (stage 2b: full fine-tune)."""
    _require(_TORCH_AVAILABLE, "torch")
    for param in model.encoder.parameters():
        param.requires_grad = True


def get_parameter_groups(
    model: Any,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-3,
) -> list[dict[str, Any]]:
    """
    Return discriminative learning-rate parameter groups for AdamW.

    Backbone parameters use a lower LR (backbone_lr) to preserve
    pretrained representations; head parameters use head_lr.
    """
    backbone_params = list(model.encoder.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params,     "lr": head_lr},
    ]
