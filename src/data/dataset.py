"""
PyTorch Dataset classes for multi-center TB CXR training.

Supports:
  - CSV-driven metadata (path, tb_label, findings_label, active_inactive,
    site, view_position, split)
  - DICOM and raster (PNG/JPEG) loading via data.preprocessing
  - On-the-fly lung-mask loading for masking / crop
  - Multi-task label bundles: TB binary + 6-class findings + active/inactive
  - Stratified split utilities (multi-center aware)
  - WeightedRandomSampler weights for class-imbalance handling

CSV schema expected::

    image_path, tb_label, findings_label, active_inactive_label,
    site, view_position, split, mask_path (optional)

    tb_label            : 0 = Normal, 1 = TB
    findings_label      : comma-separated ints (0/1 per finding class)
    active_inactive_label: 0 = Normal/Inactive, 1 = Active, -1 = Unknown
    split               : 'train' | 'val' | 'test' | 'pretrain'
"""

from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from typing import Any, Callable

_NP_AVAILABLE: bool = False
_TORCH_AVAILABLE: bool = False
_PIL_AVAILABLE: bool = False

try:
    import numpy as _np_mod  # type: ignore[import-untyped]
    _NP_AVAILABLE = True
except ImportError:
    _np_mod = None  # type: ignore[assignment]

try:
    import torch as _torch_mod  # type: ignore[import-untyped]
    from torch.utils.data import Dataset as _Dataset  # type: ignore[import-untyped]
    _TORCH_AVAILABLE = True
except ImportError:
    _torch_mod = None    # type: ignore[assignment]
    _Dataset = object    # type: ignore[assignment,misc]

try:
    from PIL import Image as _PIL_Image  # type: ignore[import-untyped]
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_Image = None    # type: ignore[assignment]


def _require(flag: bool, pkg: str) -> None:
    if not flag:
        raise ImportError(f"{pkg} is required. pip install {pkg}")


# ---------------------------------------------------------------------------
# Label schema
# ---------------------------------------------------------------------------

N_FINDINGS = 6
"""Number of TB-relevant findings: cavitation, consolidation, pleural effusion,
hilar LAD, fibrosis, nodules."""

FINDINGS_NAMES = [
    "cavitation",
    "consolidation",
    "pleural_effusion",
    "hilar_lad",
    "fibrosis",
    "nodule",
]


def parse_findings_label(raw: str, n: int = N_FINDINGS) -> list[int]:
    """Parse comma-separated finding flags (e.g. '0,1,0,0,1,0') to a list."""
    try:
        parts = [int(x.strip()) for x in raw.split(",")]
        if len(parts) != n:
            parts = (parts + [0] * n)[:n]
        return parts
    except (ValueError, AttributeError):
        return [0] * n


# ---------------------------------------------------------------------------
# Base Dataset
# ---------------------------------------------------------------------------


class TBDataset(_Dataset):  # type: ignore[misc]
    """
    Multi-task TB CXR dataset.

    Args:
        csv_path:       Path to the metadata CSV.
        image_root:     Root directory for image paths in the CSV.
        split:          Dataset split to load ('train', 'val', 'test', 'pretrain').
        transform:      Transform applied to the PIL Image before returning.
        mask_root:      Optional root for lung-mask PNG paths.
        return_mask:    Whether to return the lung mask tensor.
        target_size:    Image resize target (fallback if no transform provided).
        use_clahe:      Apply CLAHE in the dataset (vs. via transform).
        clahe_clip:     CLAHE clip limit.
        clahe_tile:     CLAHE tile grid.
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path,
        split: str = "train",
        transform: Callable | None = None,
        mask_root: str | Path | None = None,
        return_mask: bool = False,
        target_size: int = 224,
        use_clahe: bool = True,
        clahe_clip: float = 2.5,
        clahe_tile: tuple[int, int] = (8, 8),
    ) -> None:
        _require(_TORCH_AVAILABLE, "torch")
        _require(_PIL_AVAILABLE, "Pillow")

        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root) if mask_root else None
        self.split = split
        self.transform = transform
        self.return_mask = return_mask
        self.target_size = target_size
        self.use_clahe = use_clahe
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile

        self.records = self._load_csv(Path(csv_path), split)
        if not self.records:
            warnings.warn(
                f"No records found for split='{split}' in {csv_path}. "
                "Check the 'split' column.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # CSV loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_csv(csv_path: Path, split: str) -> list[dict]:
        """Read CSV and return rows matching the requested split."""
        records: list[dict] = []
        if not csv_path.exists():
            warnings.warn(f"CSV not found: {csv_path}", stacklevel=3)
            return records
        try:
            with csv_path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if row.get("split", split) == split:
                        records.append(dict(row))
        except OSError as exc:
            warnings.warn(f"Could not read {csv_path}: {exc}", stacklevel=3)
        return records

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_pil(self, path: Path) -> Any:
        """Load image from DICOM or raster file and return a uint8 PIL Image."""
        from .preprocessing import load_image  # local import avoids circular

        try:
            arr = load_image(path)  # (H, W) uint8
        except Exception as exc:
            warnings.warn(f"Image load failed for {path}: {exc}; using zeros.", stacklevel=2)
            import numpy as np  # type: ignore[import-untyped]
            arr = np.zeros((self.target_size, self.target_size), dtype=np.uint8)

        if self.use_clahe:
            from .preprocessing import apply_clahe
            arr = apply_clahe(arr, clip_limit=self.clahe_clip, tile_grid_size=self.clahe_tile)

        from PIL import Image  # type: ignore[import-untyped]
        # Convert to RGB PIL Image (transforms expect RGB)
        return Image.fromarray(arr).convert("RGB")

    def _load_mask(self, path: Path) -> Any:
        """Load binary lung mask as a uint8 numpy array."""
        _require(_NP_AVAILABLE, "numpy")
        import numpy as np  # type: ignore[import-untyped]
        import cv2  # type: ignore[import-untyped]

        try:
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise OSError(f"cv2.imread returned None for {path}")
            mask = (mask > 127).astype(np.uint8)
            return mask
        except Exception as exc:
            warnings.warn(f"Mask load failed for {path}: {exc}; using all-ones.", stacklevel=2)
            return np.ones((self.target_size, self.target_size), dtype=np.uint8)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        _require(_TORCH_AVAILABLE, "torch")
        import torch  # type: ignore[import-untyped]

        rec = self.records[idx]
        image_path = self.image_root / rec["image_path"]

        # Load and preprocess image
        pil_img = self._load_pil(image_path)

        # Apply transform (resize + normalise + augment)
        if self.transform is not None:
            img_tensor = self.transform(pil_img)
        else:
            from .augmentation import get_inference_transform  # type: ignore[import-untyped]  # noqa: PLC0415
            img_tensor = get_inference_transform(self.target_size)(pil_img)

        # Load lung mask (optional)
        mask_tensor = None
        if self.return_mask and self.mask_root is not None:
            mask_path_rel = rec.get("mask_path", "")
            if mask_path_rel:
                mask_arr = self._load_mask(self.mask_root / mask_path_rel)
                mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).float()

        # Build label bundle
        try:
            tb_label = int(rec.get("tb_label", 0))
        except (ValueError, TypeError):
            tb_label = 0

        try:
            findings_raw = rec.get("findings_label", "0,0,0,0,0,0")
            findings = torch.tensor(parse_findings_label(findings_raw), dtype=torch.float32)
        except Exception:  # noqa: BLE001
            findings = torch.zeros(N_FINDINGS, dtype=torch.float32)

        try:
            active_inactive = int(rec.get("active_inactive_label", -1))
        except (ValueError, TypeError):
            active_inactive = -1

        meta = {
            "image_path": str(image_path),
            "site": rec.get("site", "unknown"),
            "view_position": rec.get("view_position", "unknown"),
        }

        result: tuple = (img_tensor, tb_label, findings, active_inactive, meta)
        if self.return_mask and mask_tensor is not None:
            result = (img_tensor, tb_label, findings, active_inactive, meta, mask_tensor)

        return result


# ---------------------------------------------------------------------------
# Unlabelled pretraining dataset (for MoCo / DINO)
# ---------------------------------------------------------------------------


class UnlabelledCXRDataset(_Dataset):  # type: ignore[misc]
    """
    Minimal dataset for self-supervised pretraining — returns two augmented
    views of the same CXR (MoCo / SimCLR / DINO style).

    Args:
        csv_path:       Metadata CSV (needs at minimum 'image_path' column).
        image_root:     Root directory for image paths.
        view_transform: Transform applied to each view independently.
        n_views:        Number of views to return (2 for contrastive, 4 for DINO multi-crop).
        use_clahe:      Apply CLAHE before the view transform.
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path,
        view_transform: Callable | None = None,
        n_views: int = 2,
        use_clahe: bool = True,
        clahe_clip: float = 2.5,
        clahe_tile: tuple[int, int] = (8, 8),
    ) -> None:
        _require(_TORCH_AVAILABLE, "torch")
        _require(_PIL_AVAILABLE, "Pillow")

        self.image_root = Path(image_root)
        self.n_views = n_views
        self.use_clahe = use_clahe
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile

        # Default: a moderately aggressive train-style transform
        if view_transform is None:
            from .augmentation import get_train_transform  # type: ignore[import-untyped]  # noqa: PLC0415
            view_transform = get_train_transform(image_size=224)
        elif not callable(view_transform):
            raise TypeError(f"view_transform must be callable, got {type(view_transform)}")
        assert view_transform is not None and callable(view_transform)
        self.view_transform: Callable = view_transform

        self.records = self._load_csv(Path(csv_path))

    @staticmethod
    def _load_csv(csv_path: Path) -> list[dict]:
        records: list[dict] = []
        if not csv_path.exists():
            return records
        try:
            with csv_path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    records.append(dict(row))
        except OSError:
            pass
        return records

    def _load_pil(self, path: Path) -> Any:
        from .preprocessing import load_image
        try:
            arr = load_image(path)
        except Exception:  # noqa: BLE001
            import numpy as np  # type: ignore[import-untyped]
            arr = np.zeros((224, 224), dtype=np.uint8)

        if self.use_clahe:
            from .preprocessing import apply_clahe
            arr = apply_clahe(arr, clip_limit=self.clahe_clip, tile_grid_size=self.clahe_tile)

        from PIL import Image  # type: ignore[import-untyped]
        return Image.fromarray(arr).convert("RGB")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> list[Any]:
        rec = self.records[idx]
        img_path = self.image_root / rec["image_path"]
        pil_img = self._load_pil(img_path)
        transform = self.view_transform
        if not callable(transform):
            raise RuntimeError("view_transform is not callable")
        return [transform(pil_img) for _ in range(self.n_views)]


# ---------------------------------------------------------------------------
# Utility: sampler weights for class imbalance
# ---------------------------------------------------------------------------


def compute_sample_weights(
    tb_labels: list[int],
    pos_weight: float = 5.0,
) -> list[float]:
    """
    Return a per-sample weight list for ``WeightedRandomSampler``.

    TB prevalence in screening populations is 1–10 %.  By default TB-positive
    samples are up-weighted 5× to achieve a ~50/50 effective batch ratio.

    Args:
        tb_labels:  List of int labels (0=Normal, 1=TB).
        pos_weight: Multiplier applied to TB-positive samples.

    Returns:
        List of floats of the same length as ``tb_labels``.
    """
    weights = []
    for lbl in tb_labels:
        weights.append(pos_weight if lbl == 1 else 1.0)
    return weights


# ---------------------------------------------------------------------------
# Utility: stratified multi-center split
# ---------------------------------------------------------------------------


def stratified_split(
    records: list[dict],
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    seed: int = 42,
    site_column: str = "site",
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Stratify by (site, tb_label) to ensure each split contains
    proportional representation from every geographic center.

    Args:
        records:       List of metadata dicts.
        val_fraction:  Fraction of each stratum for validation.
        test_fraction: Fraction of each stratum for test.
        seed:          RNG seed for reproducibility.
        site_column:   Column name for site identifier.

    Returns:
        (train_records, val_records, test_records)
    """
    import collections
    import math

    random.seed(seed)

    # Group by (site, tb_label)
    strata: dict[tuple, list[dict]] = collections.defaultdict(list)
    for rec in records:
        key = (rec.get(site_column, "unknown"), str(rec.get("tb_label", "0")))
        strata[key].append(rec)

    train_recs: list[dict] = []
    val_recs: list[dict] = []
    test_recs: list[dict] = []

    for key, group in strata.items():
        random.shuffle(group)
        n = len(group)
        n_val = max(1, math.ceil(n * val_fraction))
        n_test = max(1, math.ceil(n * test_fraction))
        n_train = n - n_val - n_test

        if n_train <= 0:
            warnings.warn(
                f"Stratum {key!r} has only {n} samples; placing all in train.",
                stacklevel=2,
            )
            train_recs.extend(group)
            continue

        train_recs.extend(group[:n_train])
        val_recs.extend(group[n_train:n_train + n_val])
        test_recs.extend(group[n_train + n_val:])

    return train_recs, val_recs, test_recs
