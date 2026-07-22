"""
Unit tests for data/preprocessing.py.

Uses synthetic images — no real CXR data or GPU required.
Run with: pytest tests/test_preprocessing.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Stable fallbacks so names are never unbound
np: Any = None
cv2: Any = None
_DEPS_OK = False

try:
    import numpy as np          # type: ignore[import-untyped,no-redef]
    import cv2                  # type: ignore[import-untyped,no-redef]
    _DEPS_OK = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not _DEPS_OK, reason="numpy and opencv-python required"
)


@pytest.fixture()
def grey_image() -> Any:
    rng   = np.random.default_rng(42)
    base  = np.tile(np.linspace(20, 235, 256, dtype=np.uint8), (256, 1))
    noise = rng.integers(-10, 10, size=base.shape).astype(np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.fixture()
def lung_mask() -> Any:
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(mask, (128, 128), (90, 110), 0, 0, 360, 1, -1)
    return mask


class TestCLAHE:
    def test_output_dtype(self, grey_image: Any) -> None:
        from data.preprocessing import apply_clahe  # type: ignore[import-untyped]
        out = apply_clahe(grey_image)
        assert out.dtype == np.uint8

    def test_output_shape(self, grey_image: Any) -> None:
        from data.preprocessing import apply_clahe  # type: ignore[import-untyped]
        out = apply_clahe(grey_image)
        assert out.shape == grey_image.shape

    def test_contrast_improved(self, grey_image: Any) -> None:
        from data.preprocessing import apply_clahe  # type: ignore[import-untyped]
        out = apply_clahe(grey_image, clip_limit=3.0)
        assert float(out.std()) >= float(grey_image.std()) * 0.8


class TestGaussianDenoise:
    def test_zero_sigma_is_identity(self, grey_image: Any) -> None:
        from data.preprocessing import apply_gaussian_denoise  # type: ignore[import-untyped]
        out = apply_gaussian_denoise(grey_image, sigma=0)
        assert (out == grey_image).all()

    def test_nonzero_sigma_changes_image(self, grey_image: Any) -> None:
        from data.preprocessing import apply_gaussian_denoise  # type: ignore[import-untyped]
        out = apply_gaussian_denoise(grey_image, sigma=1.5)
        assert not (out == grey_image).all()


class TestLungMask:
    def test_zeros_outside_mask(self, grey_image: Any, lung_mask: Any) -> None:
        from data.preprocessing import apply_lung_mask  # type: ignore[import-untyped]
        out = apply_lung_mask(grey_image, lung_mask, fill_value=0)
        assert (out[lung_mask == 0] == 0).all()

    def test_preserves_inside_mask(self, grey_image: Any, lung_mask: Any) -> None:
        from data.preprocessing import apply_lung_mask  # type: ignore[import-untyped]
        out = apply_lung_mask(grey_image, lung_mask)
        assert (out[lung_mask == 1] == grey_image[lung_mask == 1]).all()

    def test_mismatched_mask_resized(self, grey_image: Any) -> None:
        from data.preprocessing import apply_lung_mask  # type: ignore[import-untyped]
        small = np.ones((64, 64), dtype=np.uint8)
        out   = apply_lung_mask(grey_image, small)
        assert out.shape == grey_image.shape


class TestCropToBbox:
    def test_crop_reduces_size(self, grey_image: Any, lung_mask: Any) -> None:
        from data.preprocessing import crop_to_lung_bbox  # type: ignore[import-untyped]
        cropped, _ = crop_to_lung_bbox(grey_image, lung_mask)
        assert cropped.shape[0] <= grey_image.shape[0]
        assert cropped.shape[1] <= grey_image.shape[1]

    def test_empty_mask_returns_original(self, grey_image: Any) -> None:
        from data.preprocessing import crop_to_lung_bbox  # type: ignore[import-untyped]
        empty = np.zeros((256, 256), dtype=np.uint8)
        out, _ = crop_to_lung_bbox(grey_image, empty)
        assert out.shape == grey_image.shape


class TestQC:
    def test_blank_image_fails(self) -> None:
        from data.preprocessing import passes_qc  # type: ignore[import-untyped]
        blank = np.zeros((256, 256), dtype=np.uint8)
        assert not passes_qc(blank)

    def test_normal_image_passes(self, grey_image: Any) -> None:
        from data.preprocessing import passes_qc  # type: ignore[import-untyped]
        assert passes_qc(grey_image)

    def test_small_mask_fails(self, grey_image: Any) -> None:
        from data.preprocessing import passes_qc  # type: ignore[import-untyped]
        tiny = np.zeros((256, 256), dtype=np.uint8)
        tiny[120:130, 120:130] = 1
        assert not passes_qc(grey_image, mask=tiny, min_lung_area_fraction=0.15)


class TestPreprocessCXR:
    def test_output_shape(self, grey_image: Any) -> None:
        from data.preprocessing import preprocess_cxr  # type: ignore[import-untyped]
        out = preprocess_cxr(grey_image, target_size=224)
        assert out.shape == (3, 224, 224)

    def test_output_dtype(self, grey_image: Any) -> None:
        from data.preprocessing import preprocess_cxr  # type: ignore[import-untyped]
        out = preprocess_cxr(grey_image, target_size=224)
        assert out.dtype == np.float32

    def test_with_mask(self, grey_image: Any, lung_mask: Any) -> None:
        from data.preprocessing import preprocess_cxr  # type: ignore[import-untyped]
        out = preprocess_cxr(grey_image, mask=lung_mask, target_size=224)
        assert out.shape == (3, 224, 224)

    def test_custom_size(self, grey_image: Any) -> None:
        from data.preprocessing import preprocess_cxr  # type: ignore[import-untyped]
        out = preprocess_cxr(grey_image, target_size=384)
        assert out.shape == (3, 384, 384)
