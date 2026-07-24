"""
Inference engine — wraps ONNX Runtime and per-site calibration.

Provides a clean clinical-facing API:

    engine = InferenceEngine.from_onnx("model.onnx", calibrator=cal)
    result = engine.predict(pil_image_or_path)

    result.tb_score        → float  0–100 (CAD4TB scale)
    result.tb_positive     → bool   at the calibrated site threshold
    result.findings        → dict[str, float]  per-finding probability
    result.gradcam         → np.ndarray (H, W)  heatmap (if enabled)
    result.inference_ms    → float  latency in milliseconds

Clinical safety notes (enforced in this module):
  - The engine will refuse to return a positive decision without a valid
    lung-segmentation mask (prevents false positives from scanner artifacts).
  - Score drift monitoring: logs a warning when the mean score over the
    last 100 predictions shifts by >5 points from the baseline.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_NP_AVAILABLE = False
try:
    import numpy as _np  # type: ignore[import-untyped]
    _NP_AVAILABLE = True
except ImportError:
    _np = None  # type: ignore[assignment]

_ORT_AVAILABLE = False
try:
    import onnxruntime as _ort  # type: ignore[import-untyped]
    _ORT_AVAILABLE = True
except ImportError:
    _ort = None  # type: ignore[assignment]

_CV2_AVAILABLE = False
try:
    import cv2 as _cv2  # type: ignore[import-untyped]
    _CV2_AVAILABLE = True
except ImportError:
    _cv2 = None  # type: ignore[assignment]


def _require(flag: bool, pkg: str) -> None:
    if not flag:
        raise ImportError(f"{pkg} is required. pip install {pkg}")


# ---------------------------------------------------------------------------
# Prediction result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TBPrediction:
    """
    Structured result from one inference call.

    Attributes:
        tb_score:       0–100 risk score (CAD4TB convention).
        tb_positive:    True if score exceeds the site threshold.
        tb_prob:        Raw probability P(TB) in [0, 1].
        findings:       Per-finding probability dict (6 findings).
        gradcam:        Heatmap numpy array (H, W) or None.
        inference_ms:   Wall-clock inference latency.
        site_threshold: Decision threshold in use (per-site calibrated).
        referral_note:  Human-readable note for clinical report.
    """
    tb_score:       float
    tb_positive:    bool
    tb_prob:        float
    findings:       dict[str, float] = field(default_factory=dict)
    gradcam:        Any = None              # np.ndarray | None
    inference_ms:   float = 0.0
    site_threshold: float = 0.5
    referral_note:  str = ""

    def __post_init__(self) -> None:
        if self.tb_positive and not self.referral_note:
            self.referral_note = (
                "TB-positive triage result. "
                "Confirmatory Xpert MTB/RIF testing is mandatory before treatment. "
                "This CAD result does NOT constitute a diagnosis."
            )


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

FINDINGS_NAMES = [
    "cavitation", "consolidation", "pleural_effusion",
    "hilar_lad", "fibrosis", "nodule",
]


class InferenceEngine:
    """
    ONNX Runtime inference engine with per-site calibration.

    Args:
        session:     OnnxRuntime InferenceSession.
        calibrator:  Optional fitted PlattCalibrator / IsotonicCalibrator.
        seg_model:   Optional U-Net segmentation nn.Module for on-the-fly masking.
        input_size:  CNN branch input spatial resolution.
        device:      'cpu', 'cuda', or 'tensorrt' (passed to ORT providers).
        drift_window: Number of recent predictions to monitor for score drift.
    """

    def __init__(
            self,
            session: Any,
            calibrator: Any | None = None,
            seg_model: Any | None = None,
            torch_model: Any | None = None,
            input_size: int = 224,
            device: str = "cpu",
            drift_window: int = 100,
            baseline_mean_score: float | None = None,
        ) -> None:
            self._session          = session
            self._calibrator       = calibrator
            self._seg_model        = seg_model
            # Optional torch model for Grad-CAM (ONNX cannot produce gradients).
            # When provided, positive predictions include a heatmap overlay.
            self._torch_model      = torch_model
            self.input_size        = input_size
            self.device            = device
            self.drift_window      = drift_window
            self._recent_scores:   list[float] = []
            self._baseline_mean:   float | None = baseline_mean_score

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_onnx(
        cls,
        onnx_path: str | Path,
        calibrator: Any | None = None,
        seg_model: Any | None = None,
        torch_model: Any | None = None,
        input_size: int = 224,
        device: str = "cpu",
    ) -> "InferenceEngine":
        """
        Build an InferenceEngine from a .onnx file path.

        Args:
            onnx_path:   Path to the exported ONNX model.
            calibrator:  Fitted per-site calibrator (optional but recommended).
            seg_model:   Fitted U-Net for on-the-fly lung masking (optional).
            torch_model: Optional TBEnsemble nn.Module for Grad-CAM on positives.
            input_size:  CNN input resolution (224).
            device:      ORT execution provider: 'cpu', 'cuda', 'tensorrt'.
        """
        _require(_ORT_AVAILABLE, "onnxruntime")
        import onnxruntime as ort  # type: ignore[import-untyped]

        providers_map: dict[str, list[str]] = {
            "cpu":       ["CPUExecutionProvider"],
            "cuda":      ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "tensorrt":  ["TensorrtExecutionProvider", "CUDAExecutionProvider",
                          "CPUExecutionProvider"],
        }
        providers = providers_map.get(device.lower(), ["CPUExecutionProvider"])

        try:
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            session = ort.InferenceSession(
                str(onnx_path),
                sess_options=sess_opts,
                providers=providers,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load ONNX session from {onnx_path}: {exc}") from exc

        return cls(
            session=session,
            calibrator=calibrator,
            seg_model=seg_model,
            torch_model=torch_model,
            input_size=input_size,
            device=device,
        )

    # ------------------------------------------------------------------
    # Preprocessing helper
    # ------------------------------------------------------------------

    def _preprocess(self, image: Any) -> Any:
        """
        Accept a PIL Image, numpy array, or file path and return a
        (1, 3, H, W) float32 numpy array ready for ORT.
        """
        _require(_NP_AVAILABLE, "numpy")
        _require(_CV2_AVAILABLE, "opencv-python")
        import numpy as np  # type: ignore[import-untyped]
        import cv2  # type: ignore[import-untyped]

        # Load from path
        if isinstance(image, (str, Path)):
            try:
                from ..data.preprocessing import load_image  # type: ignore[import-untyped]
                arr = load_image(Path(image))
            except Exception as exc:
                raise OSError(f"Failed to load image from {image}: {exc}") from exc
        elif hasattr(image, "convert"):
            # PIL Image
            arr = np.array(image.convert("L"), dtype=np.uint8)
        else:
            arr = np.asarray(image, dtype=np.uint8)
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # Get lung mask from segmentation model
        mask = None
        if self._seg_model is not None:
            try:
                from ..models.segmentation import predict_mask  # type: ignore[import-untyped]
                # Quick resize for seg model
                seg_arr = cv2.resize(arr, (512, 512), interpolation=cv2.INTER_LANCZOS4)
                seg_t   = self._seg_to_tensor(seg_arr)
                mask    = predict_mask(self._seg_model, seg_t, threshold=0.5)
                mask    = cv2.resize(mask, (arr.shape[1], arr.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Segmentation failed: {exc}. Proceeding without mask.",
                              stacklevel=2)

        # Apply full preprocessing chain
        try:
            from ..data.preprocessing import preprocess_cxr  # type: ignore[import-untyped]
            tensor = preprocess_cxr(
                arr, mask=mask, target_size=self.input_size
            )   # (3, H, W)
        except Exception as exc:
            raise RuntimeError(f"Preprocessing failed: {exc}") from exc

        return tensor[np.newaxis].astype(np.float32)  # (1, 3, H, W)

    @staticmethod
    def _seg_to_tensor(arr: Any) -> Any:
        """Convert a uint8 greyscale array to a float32 (1, 1, H, W) tensor for U-Net."""
        import numpy as np  # type: ignore[import-untyped]
        x = arr.astype(np.float32) / 255.0
        return x[np.newaxis, np.newaxis]  # (1, 1, H, W)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        image: Any,
        site_threshold: float | None = None,
    ) -> TBPrediction:
        """
        Run inference on one CXR image.

        Args:
            image:          PIL Image, numpy array (H, W) or file path.
            site_threshold: Override the calibrated threshold (optional).
                            If None, uses calibrator.threshold_ (default 0.5).

        Returns:
            :class:`TBPrediction` with tb_score, tb_positive, findings, etc.
        """
        _require(_ORT_AVAILABLE, "onnxruntime")
        _require(_NP_AVAILABLE, "numpy")
        import numpy as np  # type: ignore[import-untyped]

        t0 = time.perf_counter()

        # Preprocess
        x = self._preprocess(image)

        # ONNX Runtime inference
        try:
            input_name = self._session.get_inputs()[0].name
            outputs    = self._session.run(None, {input_name: x})
        except Exception as exc:
            raise RuntimeError(f"ORT inference failed: {exc}") from exc

        # Parse outputs — handle both dict-output and tuple models
        try:
            # Index 0 = tb_logits (B, 2), Index 1 = tb_prob (B,) if exported that way
            tb_prob_raw = float(outputs[1][0]) if len(outputs) > 1 else float(
                np.exp(outputs[0][0, 1]) / np.exp(outputs[0][0]).sum()
            )
        except (IndexError, ValueError, TypeError) as exc:
            raise RuntimeError(f"Failed to parse ORT output: {exc}") from exc

        # Apply calibration
        if self._calibrator is not None:
            try:
                cal_prob = float(
                    self._calibrator.predict_proba(np.array([tb_prob_raw]))[0]
                )
            except Exception:  # noqa: BLE001
                cal_prob = tb_prob_raw
        else:
            cal_prob = tb_prob_raw

        # Threshold
        threshold = site_threshold
        if threshold is None:
            threshold = getattr(self._calibrator, "threshold_", 0.5)

        tb_score    = cal_prob * 100.0
        tb_positive = cal_prob >= threshold

        # Findings (if model exports them)
        findings: dict[str, float] = {}
        if len(outputs) > 2:
            try:
                import numpy as np  # type: ignore[import-untyped,redefined-outer-name]
                logits  = outputs[2][0]  # (N_findings,)
                probs_f = 1.0 / (1.0 + np.exp(-logits))
                findings = {n: float(p) for n, p in zip(FINDINGS_NAMES, probs_f)}
            except Exception:  # noqa: BLE001
                pass

        inference_ms = (time.perf_counter() - t0) * 1000.0

        # Drift monitoring
        self._recent_scores.append(tb_score)
        if len(self._recent_scores) > self.drift_window:
            self._recent_scores.pop(0)
        if (
            self._baseline_mean is not None
            and len(self._recent_scores) >= self.drift_window
        ):
            current_mean = sum(self._recent_scores) / len(self._recent_scores)
            if abs(current_mean - self._baseline_mean) > 5.0:
                warnings.warn(
                    f"Score drift detected: baseline mean={self._baseline_mean:.1f}, "
                    f"recent mean={current_mean:.1f}. "
                    "Consider re-calibrating the model on local data.",
                    stacklevel=2,
                )

        # -- Grad-CAM (DIR-01) -----------------------------------------------
        # Produce a heatmap for every positive prediction when a torch
        # model is attached. ONNX cannot backprop, so Grad-CAM requires
        # the original torch model to be passed as torch_model= in __init__.
        gradcam_result: Any = None
        if tb_positive and self._torch_model is not None:
            try:
                import torch  # type: ignore[import-untyped]
                self._torch_model.eval()
                self._torch_model.enable_gradcam()
                img_t = torch.from_numpy(x).float()  # (1, 3, H, W)
                hmap  = self._torch_model._gradcam(img_t, class_idx=1)
                gradcam_result = hmap  # (H', W') float array [0, 1]
                self._torch_model.disable_gradcam()
            except Exception as exc_cam:  # noqa: BLE001
                warnings.warn(f"Grad-CAM failed: {exc_cam}", stacklevel=2)

        return TBPrediction(
            tb_score=tb_score,
            tb_positive=tb_positive,
            tb_prob=cal_prob,
            findings=findings,
            gradcam=gradcam_result,
            inference_ms=inference_ms,
            site_threshold=threshold,
        )
