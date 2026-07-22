"""
Preprocessing pipeline for chest X-ray images.

Implements the recommended steps from the research plan:
  1. DICOM decoding with ModalityLUT/VOILUT and 8-bit conversion
  2. View-type detection and lateral rejection
  3. CLAHE contrast enhancement (clip=2.5, tile 8×8)
  4. Gaussian denoising
  5. Lung-field masking (applies the U-Net mask externally)
  6. Resize + normalise to ImageNet stats
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import cv2  # type: ignore[import-untyped]
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    import numpy as np  # type: ignore[import-untyped]
    _NP_AVAILABLE = True
except ImportError:
    _NP_AVAILABLE = False

try:
    import pydicom  # type: ignore[import-untyped]
    _DICOM_AVAILABLE = True
except ImportError:
    _DICOM_AVAILABLE = False

if TYPE_CHECKING:
    import pydicom  # noqa: F811

try:
    from PIL import Image  # type: ignore[import-untyped]
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


def _require(flag: bool, pkg: str) -> None:
    if not flag:
        raise ImportError(f"{pkg} is required. Install with: pip install {pkg}")


# ---------------------------------------------------------------------------
# DICOM helpers
# ---------------------------------------------------------------------------


def load_dicom(path: str | Path) -> "np.ndarray":
    """
    Load a DICOM file and return a 2-D uint8 numpy array (H, W).

    Applies the full display pipeline:
      ModalityLUT (rescale slope/intercept) → VOILUT (window centre/width) → 8-bit.
    """
    _require(_DICOM_AVAILABLE, "pydicom")
    _require(_NP_AVAILABLE, "numpy")

    try:
        import pydicom as _pydicom  # type: ignore[import-untyped]
        import numpy as _np  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("pydicom and numpy are required for DICOM loading.") from exc

    try:
        dcm = _pydicom.dcmread(str(path))
    except Exception as exc:
        raise OSError(f"Failed to read DICOM file {path}: {exc}") from exc

    try:
        pixel_array = dcm.pixel_array.astype(_np.float32)
    except Exception as exc:
        raise ValueError(f"Cannot extract pixel data from {path}: {exc}") from exc

    # Apply ModalityLUT (rescale slope/intercept)
    try:
        slope = float(getattr(dcm, "RescaleSlope", 1))
        intercept = float(getattr(dcm, "RescaleIntercept", 0))
        pixel_array = pixel_array * slope + intercept
    except (TypeError, ValueError) as exc:
        warnings.warn(f"Could not apply ModalityLUT for {path}: {exc}", stacklevel=2)

    # Apply VOILUT (window/level)
    try:
        lo: float = float(pixel_array.min())
        hi: float = float(pixel_array.max())
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Cannot compute pixel range for {path}: {exc}") from exc
    try:
        if hasattr(dcm, "WindowCenter") and hasattr(dcm, "WindowWidth"):
            wc_raw = dcm.WindowCenter
            ww_raw = dcm.WindowWidth
            wc = float(
                wc_raw[0] if hasattr(wc_raw, "__iter__") else wc_raw
            )
            ww = float(
                ww_raw[0] if hasattr(ww_raw, "__iter__") else ww_raw
            )
            lo = wc - ww / 2.0
            hi = wc + ww / 2.0
            pixel_array = _np.clip(pixel_array, lo, hi)
    except (TypeError, ValueError, AttributeError) as exc:
        warnings.warn(f"Could not apply VOILUT for {path}: {exc}; using full range.", stacklevel=2)

    # Photometric interpretation — invert if MONOCHROME1
    try:
        photometric = str(getattr(dcm, "PhotometricInterpretation", "MONOCHROME2"))
        if photometric.strip().upper() == "MONOCHROME1":
            pixel_array = hi - pixel_array + lo
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Could not determine photometric interpretation: {exc}", stacklevel=2)

    # Scale to [0, 255]
    try:
        span = hi - lo
        if span > 0:
            pixel_array = (pixel_array - lo) / span * 255.0
        pixel_array = _np.clip(pixel_array, 0, 255).astype(_np.uint8)
    except Exception as exc:
        raise ValueError(f"Failed to scale pixel array to uint8: {exc}") from exc

    return pixel_array


def get_view_position(path: str | Path) -> str | None:
    """
    Return the DICOM ViewPosition tag (e.g. 'PA', 'AP', 'LL', 'RL').
    Returns None if tag is absent or file is not a DICOM.
    """
    if not _DICOM_AVAILABLE:
        return None
    try:
        import pydicom as _pydicom  # type: ignore[import-untyped]
        dcm = _pydicom.dcmread(str(path), stop_before_pixels=True)
        return str(getattr(dcm, "ViewPosition", "")).strip().upper() or None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Image loading (DICOM + raster)
# ---------------------------------------------------------------------------


def load_image(path: str | Path) -> "np.ndarray":
    """
    Load any CXR image (DICOM, PNG, JPEG) as a 2-D uint8 array (H, W).
    PNG/JPEG are converted to greyscale.
    """
    _require(_NP_AVAILABLE, "numpy")
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".dcm", ""):
        try:
            return load_dicom(path)
        except Exception:  # noqa: BLE001
            pass  # fall through to PIL

    _require(_PIL_AVAILABLE, "Pillow")
    try:
        from PIL import Image as _Image  # type: ignore[import-untyped]
        import numpy as _np  # type: ignore[import-untyped]
        img = _Image.open(str(path)).convert("L")
        return _np.array(img, dtype=_np.uint8)
    except Exception as exc:
        raise OSError(f"Failed to load image {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# CLAHE contrast enhancement
# ---------------------------------------------------------------------------


def apply_clahe(
    image: "np.ndarray",
    clip_limit: float = 2.5,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> "np.ndarray":
    """
    Apply Contrast-Limited Adaptive Histogram Equalisation (CLAHE).

    This is the single most-cited preprocessing step for TB CXR — it
    substantially improves the visibility of cavitations, consolidations,
    and pleural effusions.

    Args:
        image:          2-D uint8 greyscale image.
        clip_limit:     CLAHE clip limit (2.0–3.0 per plan).
        tile_grid_size: Tile size for local histogram computation.

    Returns:
        CLAHE-enhanced uint8 image of the same shape.
    """
    _require(_CV2_AVAILABLE, "opencv-python")
    try:
        clahe = cv2.createCLAHE(  # type: ignore[name-defined]
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size,
        )
        return clahe.apply(image)
    except Exception as exc:
        warnings.warn(f"CLAHE failed ({exc}), returning original image.", stacklevel=2)
        return image


# ---------------------------------------------------------------------------
# Gaussian denoising
# ---------------------------------------------------------------------------


def apply_gaussian_denoise(
    image: "np.ndarray",
    sigma: float = 0.5,
) -> "np.ndarray":
    """
    Mild Gaussian low-pass filter to reduce digitisation noise.

    Sigma ≤ 1.0 preserves fine texture (cavitation walls, nodule borders).
    """
    _require(_CV2_AVAILABLE, "opencv-python")
    if sigma <= 0:
        return image
    try:
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)  # type: ignore[name-defined]
    except Exception as exc:
        warnings.warn(f"Gaussian blur failed ({exc}), returning original.", stacklevel=2)
        return image


# ---------------------------------------------------------------------------
# Lung-field masking
# ---------------------------------------------------------------------------


def apply_lung_mask(
    image: "np.ndarray",
    mask: "np.ndarray",
    fill_value: int = 0,
) -> "np.ndarray":
    """
    Zero-out extra-thoracic regions using a binary lung mask.

    Suppresses shortcut features: text overlays, jewellery, pacemakers,
    and scanner halos that the model would otherwise latch onto.

    Args:
        image:      2-D or 3-D float/uint8 image (H, W) or (H, W, C).
        mask:       Binary 2-D mask (H, W) — 1 inside lungs, 0 outside.
        fill_value: Value assigned to masked-out pixels.

    Returns:
        Masked image of the same shape and dtype.
    """
    _require(_CV2_AVAILABLE, "opencv-python")
    _require(_NP_AVAILABLE, "numpy")
    try:
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(  # type: ignore[name-defined]
                mask.astype(np.uint8),  # type: ignore[name-defined]
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,  # type: ignore[name-defined]
            )
        out = image.copy()
        if image.ndim == 3:
            out[mask == 0, :] = fill_value
        else:
            out[mask == 0] = fill_value
        return out
    except Exception as exc:
        warnings.warn(f"Lung mask application failed ({exc}), returning original.", stacklevel=2)
        return image


def crop_to_lung_bbox(
    image: "np.ndarray",
    mask: "np.ndarray",
    padding: float = 0.05,
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Crop image (and mask) to the tight bounding box of the lung mask,
    with a small fractional padding to avoid clipping at the borders.

    Returns (cropped_image, cropped_mask).
    """
    _require(_NP_AVAILABLE, "numpy")
    try:
        rows = np.any(mask > 0, axis=1)  # type: ignore[name-defined]
        cols = np.any(mask > 0, axis=0)  # type: ignore[name-defined]
        if not rows.any() or not cols.any():
            return image, mask

        r_min, r_max = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])  # type: ignore[name-defined]
        c_min, c_max = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])  # type: ignore[name-defined]

        pad_r = int((r_max - r_min) * padding)
        pad_c = int((c_max - c_min) * padding)
        H, W = image.shape[:2]

        r_min = max(0, r_min - pad_r)
        r_max = min(H, r_max + pad_r + 1)
        c_min = max(0, c_min - pad_c)
        c_max = min(W, c_max + pad_c + 1)

        return image[r_min:r_max, c_min:c_max], mask[r_min:r_max, c_min:c_max]
    except Exception as exc:
        warnings.warn(f"Bounding-box crop failed ({exc}), returning original.", stacklevel=2)
        return image, mask


# ---------------------------------------------------------------------------
# QC check
# ---------------------------------------------------------------------------


def passes_qc(
    image: "np.ndarray",
    mask: "np.ndarray | None" = None,
    min_lung_area_fraction: float = 0.15,
) -> bool:
    """
    Basic quality-control gate.

    Returns False for:
    - Near-blank images (std < 10)
    - Severely under-exposed images (mean < 20)
    - Lung mask covers <15 % of image area (severe cropping / wrong view)
    """
    _require(_NP_AVAILABLE, "numpy")
    try:
        if float(image.std()) < 10.0:
            return False
        if float(image.mean()) < 20.0:
            return False
        if mask is not None:
            lung_fraction = float(mask.mean())
            if lung_fraction < min_lung_area_fraction:
                return False
    except Exception:  # noqa: BLE001
        return False
    return True


# ---------------------------------------------------------------------------
# Full preprocessing function
# ---------------------------------------------------------------------------


def preprocess_cxr(
    image: "np.ndarray",
    mask: "np.ndarray | None" = None,
    target_size: int = 224,
    clahe_clip: float = 2.5,
    clahe_tile: tuple[int, int] = (8, 8),
    gaussian_sigma: float = 0.5,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> "np.ndarray":
    """
    Full preprocessing chain: CLAHE → denoise → mask → crop → resize → normalise.

    Args:
        image:       Raw 2-D uint8 greyscale image.
        mask:        Binary 2-D lung mask (same H×W as image). Optional.
        target_size: Output spatial resolution (square).
        clahe_clip:  CLAHE clip limit.
        clahe_tile:  CLAHE tile grid.
        gaussian_sigma: Gaussian blur sigma.
        mean/std:    Per-channel normalisation constants (ImageNet defaults).

    Returns:
        Float32 numpy array of shape (3, target_size, target_size),
        normalised and ready for a PyTorch model.
    """
    _require(_CV2_AVAILABLE, "opencv-python")
    _require(_NP_AVAILABLE, "numpy")

    import numpy as _np  # type: ignore[import-untyped]

    # 1. CLAHE
    enhanced = apply_clahe(image, clip_limit=clahe_clip, tile_grid_size=clahe_tile)

    # 2. Gaussian denoise
    denoised = apply_gaussian_denoise(enhanced, sigma=gaussian_sigma)

    # 3. Apply lung mask + crop to lung bounding box
    if mask is not None:
        denoised = apply_lung_mask(denoised, mask)
        denoised, _ = crop_to_lung_bbox(denoised, mask)

    # 4. Resize
    try:
        resized = cv2.resize(  # type: ignore[name-defined]
            denoised,
            (target_size, target_size),
            interpolation=cv2.INTER_LANCZOS4,  # type: ignore[name-defined]
        )
    except Exception as exc:
        raise RuntimeError(f"Resize to {target_size}×{target_size} failed: {exc}") from exc

    # 5. Convert to 3-channel float [0, 1]
    try:
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB).astype(_np.float32) / 255.0  # type: ignore[name-defined]
    except Exception as exc:
        raise RuntimeError(f"Grayscale-to-RGB conversion failed: {exc}") from exc

    # 6. Normalise with ImageNet stats
    rgb = (rgb - _np.array(mean, dtype=_np.float32)) / _np.array(std, dtype=_np.float32)

    # 7. HWC → CHW for PyTorch
    return rgb.transpose(2, 0, 1).astype(_np.float32)  # (3, H, W)
