"""
Training-time augmentation policy for TB CXR.

Key constraint from the research plan:
  - NO horizontal flip — mirroring a chest X-ray places the heart on the
    right side, producing an anatomically impossible image that corrupts the
    model's spatial priors for apical/basal disease distribution.

Implemented transforms (all safe for CXR):
  - RandomAffine  : small rotation (±10°), translation, scale
  - ColorJitter   : brightness + contrast (no saturation/hue — greyscale)
  - CutMix        : within-lung-field random patch mixing
  - MixUp         : soft label interpolation between two samples
  - RandomErasing : simulate occluded regions / over-exposed patches
"""

from __future__ import annotations

import random
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Soft-optional dependency flags — set unconditionally before any import
# ---------------------------------------------------------------------------
_NP_AVAILABLE: bool = False
_TORCH_AVAILABLE: bool = False

try:
    import numpy as _np_mod  # type: ignore[import-untyped]
    _NP_AVAILABLE = True
except ImportError:
    _np_mod = None  # type: ignore[assignment]

try:
    import torch as _torch_mod  # type: ignore[import-untyped]
    import torchvision.transforms as _T_mod  # type: ignore[import-untyped]
    _TORCH_AVAILABLE = True
except ImportError:
    _torch_mod = None  # type: ignore[assignment]
    _T_mod = None      # type: ignore[assignment]


def _require(flag: bool, pkg: str) -> None:
    if not flag:
        raise ImportError(f"{pkg} is required. pip install {pkg}")


# ---------------------------------------------------------------------------
# Standard deterministic transforms (used at inference too)
# ---------------------------------------------------------------------------


def get_inference_transform(
    image_size: int = 224,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Any:
    """
    Minimal deterministic transform for validation / inference.

    Expects a PIL Image and returns a float tensor (3, H, W) normalised
    to ImageNet stats.
    """
    _require(_TORCH_AVAILABLE, "torchvision")
    import torchvision.transforms as T  # type: ignore[import-untyped]

    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


# ---------------------------------------------------------------------------
# Training augmentation pipeline
# ---------------------------------------------------------------------------


def get_train_transform(
    image_size: int = 224,
    rotation_degrees: float = 10.0,
    translate_fraction: float = 0.05,
    scale_range: tuple[float, float] = (0.85, 1.15),
    brightness_jitter: float = 0.2,
    contrast_jitter: float = 0.2,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    use_random_erasing: bool = True,
) -> Any:
    """
    Training-time augmentation — NO horizontal or vertical flip.

    Args:
        image_size:         Square output resolution.
        rotation_degrees:   Max rotation angle (±degrees).
        translate_fraction: Max translate as fraction of image size.
        scale_range:        (min_scale, max_scale) for RandomAffine.
        brightness_jitter:  ColorJitter brightness magnitude.
        contrast_jitter:    ColorJitter contrast magnitude.
        mean / std:         Normalisation constants.
        use_random_erasing: Drop small random rectangular patches.

    Returns:
        A ``torchvision.transforms.Compose`` pipeline.
    """
    _require(_TORCH_AVAILABLE, "torchvision")
    import torchvision.transforms as T  # type: ignore[import-untyped]

    transforms: list[Callable] = [
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.LANCZOS),
        # Geometric — never flip
        T.RandomAffine(
            degrees=rotation_degrees,
            translate=(translate_fraction, translate_fraction),
            scale=scale_range,
            interpolation=T.InterpolationMode.BILINEAR,
            fill=0,
        ),
        # Photometric (greyscale-safe: no saturation or hue)
        T.ColorJitter(brightness=brightness_jitter, contrast=contrast_jitter),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]

    if use_random_erasing:
        transforms.append(
            T.RandomErasing(p=0.2, scale=(0.02, 0.10), ratio=(0.5, 2.0), value=0),
        )

    return T.Compose(transforms)


# ---------------------------------------------------------------------------
# CutMix  (within-lung-field)
# ---------------------------------------------------------------------------


def cutmix_batch(
    images: Any,
    labels: Any,
    alpha: float = 1.0,
    lung_masks: Any | None = None,
) -> tuple[Any, Any, Any, float]:
    """
    Apply CutMix to a batch of images.

    If ``lung_masks`` is provided, the cut region is constrained to the
    bounding box of the lung mask, preventing anatomically meaningless
    patches from being pasted into non-lung regions.

    Args:
        images:     Float tensor (B, C, H, W).
        labels:     Long tensor (B,) or float tensor (B, n_classes).
        alpha:      Beta-distribution concentration parameter.
        lung_masks: Optional binary masks (B, 1, H, W).

    Returns:
        (mixed_images, labels_a, labels_b, lambda)
    """
    _require(_TORCH_AVAILABLE, "torch")
    _require(_NP_AVAILABLE, "numpy")

    import numpy as np  # type: ignore[import-untyped]
    import torch  # type: ignore[import-untyped]

    B, C, H, W = images.shape

    try:
        lam: float = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    except (ValueError, TypeError):
        lam = 1.0

    rand_idx = torch.randperm(B, device=images.device)
    shuffled = images[rand_idx]
    labels_b = labels[rand_idx]

    try:
        cut_ratio = float(np.sqrt(max(0.0, 1.0 - lam)))
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"CutMix geometry computation failed: {exc}") from exc

    if lung_masks is not None:
        try:
            mask_np = lung_masks[0, 0].cpu().numpy()
            rows = np.any(mask_np > 0, axis=1)
            cols = np.any(mask_np > 0, axis=0)
            if rows.any() and cols.any():
                r0 = int(np.where(rows)[0][0])
                r1 = int(np.where(rows)[0][-1])
                c0 = int(np.where(cols)[0][0])
                c1 = int(np.where(cols)[0][-1])
                cy = random.randint(r0, max(r0 + 1, r1))
                cx = random.randint(c0, max(c0 + 1, c1))
            else:
                cy = random.randint(0, H)
                cx = random.randint(0, W)
        except Exception:  # noqa: BLE001
            cy = random.randint(0, H)
            cx = random.randint(0, W)
    else:
        cy = random.randint(0, H)
        cx = random.randint(0, W)

    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = shuffled[:, :, y1:y2, x1:x2]

    actual_lam = 1.0 - (y2 - y1) * (x2 - x1) / max(H * W, 1)
    return mixed, labels, labels_b, actual_lam


# ---------------------------------------------------------------------------
# MixUp
# ---------------------------------------------------------------------------


def mixup_batch(
    images: Any,
    labels: Any,
    alpha: float = 0.4,
    n_classes: int | None = None,
) -> tuple[Any, Any, Any, float]:
    """
    MixUp: linearly interpolate images and (optionally one-hot) labels.

    Args:
        images:    Float tensor (B, C, H, W).
        labels:    Long tensor (B,) for hard labels, or (B, n_classes) soft.
        alpha:     Beta concentration parameter.
        n_classes: Required when labels are hard (long) to convert to one-hot.

    Returns:
        (mixed_images, labels_a, labels_b, lambda)
    """
    _require(_TORCH_AVAILABLE, "torch")
    _require(_NP_AVAILABLE, "numpy")

    import numpy as np  # type: ignore[import-untyped]
    import torch  # type: ignore[import-untyped]

    try:
        lam: float = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    except (ValueError, TypeError):
        lam = 1.0

    B = images.shape[0]
    rand_idx = torch.randperm(B, device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[rand_idx]

    if labels.dtype == torch.long and n_classes is not None:
        one_hot = torch.zeros(B, n_classes, device=labels.device, dtype=torch.float32)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        labels_b = one_hot[rand_idx]
        return mixed_images, one_hot, labels_b, lam

    return mixed_images, labels, labels[rand_idx], lam


# ---------------------------------------------------------------------------
# Collate helper for DataLoader
# ---------------------------------------------------------------------------


class CutMixMixUpCollator:
    """
    Drop-in DataLoader ``collate_fn`` that randomly applies CutMix or MixUp
    to each batch with configurable probability.

    Usage::

        loader = DataLoader(
            dataset,
            collate_fn=CutMixMixUpCollator(n_classes=2),
        )
    """

    def __init__(
        self,
        n_classes: int = 2,
        cutmix_alpha: float = 1.0,
        mixup_alpha: float = 0.4,
        cutmix_prob: float = 0.5,
        mixup_prob: float = 0.5,
    ) -> None:
        self.n_classes = n_classes
        self.cutmix_alpha = cutmix_alpha
        self.mixup_alpha = mixup_alpha
        self.cutmix_prob = cutmix_prob
        self.mixup_prob = mixup_prob

    def __call__(self, batch: list[tuple]) -> tuple[Any, dict]:
        _require(_TORCH_AVAILABLE, "torch")
        import torch  # type: ignore[import-untyped]

        images = torch.stack([b[0] for b in batch])
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)

        r = random.random()
        if r < self.cutmix_prob:
            images, la, lb, lam = cutmix_batch(images, labels, alpha=self.cutmix_alpha)
            return images, {"labels_a": la, "labels_b": lb, "lam": lam, "mode": "cutmix"}

        if r < self.cutmix_prob + self.mixup_prob:
            images, la, lb, lam = mixup_batch(
                images, labels, alpha=self.mixup_alpha, n_classes=self.n_classes
            )
            return images, {"labels_a": la, "labels_b": lb, "lam": lam, "mode": "mixup"}

        # No augmentation — return standard labels dict
        one_hot = torch.zeros(len(labels), self.n_classes, dtype=torch.float32)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        return images, {"labels_a": one_hot, "labels_b": one_hot, "lam": 1.0, "mode": "none"}
