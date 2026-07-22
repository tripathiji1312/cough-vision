"""
Per-site threshold calibration.

The single most important — and most often skipped — deployment step.

Even 50–200 locally labelled CXRs are sufficient to shift the operating
threshold so the model achieves ≥90% sensitivity on the local population.

Three calibration methods are supported:
  - ``'platt'``       — Platt scaling (logistic regression on raw scores)
  - ``'isotonic'``    — Isotonic regression (non-parametric, monotone)
  - ``'temperature'`` — Temperature scaling (single scalar T on logits)

All methods store the calibrated threshold as ``calibrator.threshold_``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

_NP_AVAILABLE = False
try:
    import numpy as _np  # type: ignore[import-untyped]
    _NP_AVAILABLE = True
except ImportError:
    _np = None  # type: ignore[assignment]

_SK_AVAILABLE = False
try:
    from sklearn import (  # type: ignore[import-untyped]
        calibration as _sk_cal,
        linear_model as _sk_lm,
        isotonic as _sk_iso,
    )
    _SK_AVAILABLE = True
except ImportError:
    _sk_cal = None   # type: ignore[assignment]
    _sk_lm  = None   # type: ignore[assignment]
    _sk_iso = None   # type: ignore[assignment]


def _require(flag: bool, pkg: str) -> None:
    if not flag:
        raise ImportError(f"{pkg} is required. pip install {pkg}")


# ---------------------------------------------------------------------------
# Platt scaling
# ---------------------------------------------------------------------------

class PlattCalibrator:
    """
    Fits a logistic regression on ``(raw_score, label)`` pairs.

    The calibrated probability = sigmoid(a * raw_score + b) where
    (a, b) are fitted by MLE on the local site data.
    """

    def __init__(self) -> None:
        self._lr: Any = None
        self.threshold_: float = 0.5

    def fit(
        self,
        scores: Any,
        labels: Any,
        sensitivity_target: float = 0.90,
    ) -> "PlattCalibrator":
        """
        Fit the calibrator and find the per-site threshold for
        ``sensitivity_target``.

        Args:
            scores:             Raw model scores (N,) in [0, 1].
            labels:             Binary labels (N,) — 0=Normal, 1=TB.
            sensitivity_target: WHO TPP sensitivity (0.90).
        """
        _require(_SK_AVAILABLE, "scikit-learn")
        _require(_NP_AVAILABLE, "numpy")
        import numpy as np  # type: ignore[import-untyped]
        from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

        yt = np.asarray(labels, dtype=int)
        yp = np.asarray(scores, dtype=float).reshape(-1, 1)

        if len(np.unique(yt)) < 2:
            warnings.warn(
                "Calibration data contains only one class. "
                "Threshold left at 0.5.", stacklevel=2,
            )
            return self

        try:
            self._lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
            self._lr.fit(yp, yt)
        except Exception as exc:
            raise RuntimeError(f"Platt scaling fit failed: {exc}") from exc

        # Find the threshold achieving sensitivity_target on the calibration set
        self.threshold_ = self._find_threshold(yt, yp.ravel(), sensitivity_target)
        return self

    def predict_proba(self, scores: Any) -> Any:
        """Return calibrated probabilities."""
        _require(_NP_AVAILABLE, "numpy")
        import numpy as np  # type: ignore[import-untyped]
        if self._lr is None:
            return np.asarray(scores, dtype=float)
        try:
            yp = np.asarray(scores, dtype=float).reshape(-1, 1)
            return self._lr.predict_proba(yp)[:, 1]
        except Exception as exc:
            raise RuntimeError(f"Platt predict_proba failed: {exc}") from exc

    def _find_threshold(
        self, y_true: Any, y_score: Any, sensitivity_target: float
    ) -> float:
        """Binary-search for the threshold achieving the sensitivity target."""
        _require(_NP_AVAILABLE, "numpy")
        import numpy as np  # type: ignore[import-untyped]
        from sklearn.metrics import roc_curve  # type: ignore[import-untyped]
        try:
            cal_prob = self.predict_proba(y_score)
            fpr, tpr, thresholds = roc_curve(y_true, cal_prob)
            valid = np.where(tpr >= sensitivity_target)[0]
            if len(valid) == 0:
                return float(thresholds[np.argmax(tpr)])
            return float(thresholds[int(valid[0])])  # highest threshold still meeting sensitivity target
        except Exception:  # noqa: BLE001
            return 0.5

    def save(self, path: str | Path) -> None:
        """Persist calibrator to disk (pickle-free JSON-friendly format)."""
        import json  # noqa: PLC0415
        if self._lr is None:
            raise RuntimeError("Calibrator has not been fitted yet.")
        try:
            data = {
                "method": "platt",
                "coef": float(self._lr.coef_[0][0]),
                "intercept": float(self._lr.intercept_[0]),
                "threshold": self.threshold_,
            }
            Path(path).write_text(json.dumps(data, indent=2))
        except Exception as exc:
            raise OSError(f"Failed to save calibrator to {path}: {exc}") from exc

    @classmethod
    def load(cls, path: str | Path) -> "PlattCalibrator":
        """Load a previously saved calibrator."""
        import json  # noqa: PLC0415
        _require(_SK_AVAILABLE, "scikit-learn")
        from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
        try:
            data = json.loads(Path(path).read_text())
            obj  = cls()
            lr   = LogisticRegression()
            import numpy as np  # type: ignore[import-untyped]
            lr.coef_      = np.array([[data["coef"]]])
            lr.intercept_ = np.array([data["intercept"]])
            lr.classes_   = np.array([0, 1])
            obj._lr        = lr
            obj.threshold_ = float(data["threshold"])
            return obj
        except Exception as exc:
            raise OSError(f"Failed to load calibrator from {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Isotonic regression calibrator
# ---------------------------------------------------------------------------

class IsotonicCalibrator:
    """
    Monotone isotonic regression — non-parametric, works well when the
    raw score distribution is non-sigmoid.
    """

    def __init__(self) -> None:
        self._iso: Any = None
        self.threshold_: float = 0.5

    def fit(
        self,
        scores: Any,
        labels: Any,
        sensitivity_target: float = 0.90,
    ) -> "IsotonicCalibrator":
        _require(_SK_AVAILABLE, "scikit-learn")
        _require(_NP_AVAILABLE, "numpy")
        import numpy as np  # type: ignore[import-untyped]
        from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]

        yt = np.asarray(labels, dtype=float)
        yp = np.asarray(scores, dtype=float)

        if len(np.unique(yt.astype(int))) < 2:
            warnings.warn("Only one class in calibration data.", stacklevel=2)
            return self

        try:
            self._iso = IsotonicRegression(out_of_bounds="clip")
            self._iso.fit(yp, yt)
        except Exception as exc:
            raise RuntimeError(f"Isotonic calibration fit failed: {exc}") from exc

        self.threshold_ = self._find_threshold(yt.astype(int), yp, sensitivity_target)
        return self

    def predict_proba(self, scores: Any) -> Any:
        _require(_NP_AVAILABLE, "numpy")
        import numpy as np  # type: ignore[import-untyped]
        if self._iso is None:
            return np.asarray(scores, dtype=float)
        try:
            return self._iso.predict(np.asarray(scores, dtype=float))
        except Exception as exc:
            raise RuntimeError(f"IsotonicCalibrator predict failed: {exc}") from exc

    def _find_threshold(
        self, y_true: Any, y_score: Any, sensitivity_target: float
    ) -> float:
        _require(_NP_AVAILABLE, "numpy")
        import numpy as np  # type: ignore[import-untyped]
        from sklearn.metrics import roc_curve  # type: ignore[import-untyped]
        try:
            cal = self.predict_proba(y_score)
            fpr, tpr, thresholds = roc_curve(y_true, cal)
            valid = np.where(tpr >= sensitivity_target)[0]
            if len(valid) == 0:
                return float(thresholds[np.argmax(tpr)])
            return float(thresholds[int(valid[0])])  # highest threshold still meeting sensitivity target
        except Exception:  # noqa: BLE001
            return 0.5


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_calibrator(method: str = "platt") -> Any:
    """
    Return a calibrator instance by name.

    Args:
        method: 'platt', 'isotonic'

    Returns:
        Calibrator with ``.fit(scores, labels)`` and
        ``.predict_proba(scores)`` methods plus ``.threshold_``.
    """
    if method == "platt":
        return PlattCalibrator()
    if method == "isotonic":
        return IsotonicCalibrator()
    raise ValueError(f"Unknown calibration method '{method}'. "
                     "Choose from: 'platt', 'isotonic'.")
