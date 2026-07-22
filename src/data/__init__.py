"""
data package — CXR preprocessing, augmentation, and dataset utilities.

All symbols are re-exported here so callers can do:

    from data import TBDataset, preprocess_cxr, get_train_transform
"""

from __future__ import annotations

# Re-export preprocessing (only hard dep is numpy + cv2 — always available)
from .preprocessing import (  # type: ignore[import-untyped]
    apply_clahe,
    apply_gaussian_denoise,
    apply_lung_mask,
    crop_to_lung_bbox,
    get_view_position,
    load_dicom,
    load_image,
    passes_qc,
    preprocess_cxr,
)

# Augmentation and dataset require torch/torchvision — guard gracefully
try:
    from .augmentation import (  # type: ignore[import-untyped]
        CutMixMixUpCollator,
        cutmix_batch,
        get_inference_transform,
        get_train_transform,
        mixup_batch,
    )
except ImportError:
    pass  # torch not installed — augmentation unavailable

try:
    from .dataset import (  # type: ignore[import-untyped]
        FINDINGS_NAMES,
        N_FINDINGS,
        TBDataset,
        UnlabelledCXRDataset,
        compute_sample_weights,
        parse_findings_label,
        stratified_split,
    )
except ImportError:
    pass  # torch not installed — dataset unavailable

__all__ = [
    # preprocessing
    "load_image", "load_dicom", "get_view_position",
    "apply_clahe", "apply_gaussian_denoise",
    "apply_lung_mask", "crop_to_lung_bbox",
    "passes_qc", "preprocess_cxr",
    # augmentation
    "get_train_transform", "get_inference_transform",
    "cutmix_batch", "mixup_batch", "CutMixMixUpCollator",
    # dataset
    "TBDataset", "UnlabelledCXRDataset",
    "compute_sample_weights", "stratified_split",
    "parse_findings_label", "N_FINDINGS", "FINDINGS_NAMES",
]
