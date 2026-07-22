"""
Unit tests for evaluation/metrics.py and evaluation/calibration.py.

Uses synthetic score distributions — no real CXR data or GPU required.
Run with: pytest tests/test_evaluation.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Stable fallbacks — names never unbound regardless of install status
np: Any = None
roc_auc_score: Any = None
_DEPS_OK = False

try:
    import numpy as np                          # type: ignore[import-untyped,no-redef]
    from sklearn.metrics import roc_auc_score   # type: ignore[import-untyped,no-redef]
    _DEPS_OK = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not _DEPS_OK, reason="numpy and scikit-learn required"
)


@pytest.fixture()
def perfect_scores() -> tuple[Any, Any]:
    n_pos  = 80
    n_neg  = 320
    y_true = np.array([1] * n_pos + [0] * n_neg, dtype=int)
    y_prob = np.array([1.0] * n_pos + [0.0] * n_neg, dtype=float)
    return y_true, y_prob


@pytest.fixture()
def noisy_scores() -> tuple[Any, Any]:
    rng   = np.random.default_rng(42)
    pos   = rng.beta(8, 2, 50)
    neg   = rng.beta(2, 8, 200)
    y_true = np.array([1] * 50 + [0] * 200, dtype=int)
    y_prob = np.concatenate([pos, neg])
    return y_true, y_prob


class TestAUCMetrics:
    def test_perfect_auc_roc(self, perfect_scores: tuple) -> None:
        from evaluation.metrics import auc_metrics  # type: ignore[import-untyped]
        yt, yp = perfect_scores
        assert auc_metrics(yt, yp)["auc_roc"] == pytest.approx(1.0, abs=1e-3)

    def test_perfect_auc_pr(self, perfect_scores: tuple) -> None:
        from evaluation.metrics import auc_metrics  # type: ignore[import-untyped]
        yt, yp = perfect_scores
        assert auc_metrics(yt, yp)["auc_pr"] == pytest.approx(1.0, abs=1e-3)

    def test_noisy_auc_reasonable(self, noisy_scores: tuple) -> None:
        from evaluation.metrics import auc_metrics  # type: ignore[import-untyped]
        yt, yp = noisy_scores
        assert 0.7 < auc_metrics(yt, yp)["auc_roc"] < 1.0


class TestClassificationMetrics:
    def test_perfect_sensitivity_specificity(self, perfect_scores: tuple) -> None:
        from evaluation.metrics import classification_metrics  # type: ignore[import-untyped]
        yt, yp = perfect_scores
        m = classification_metrics(yt, yp, threshold=0.5)
        assert m["sensitivity"] == pytest.approx(1.0)
        assert m["specificity"] == pytest.approx(1.0)

    def test_keys_present(self, noisy_scores: tuple) -> None:
        from evaluation.metrics import classification_metrics  # type: ignore[import-untyped]
        yt, yp = noisy_scores
        m = classification_metrics(yt, yp)
        for key in ("sensitivity", "specificity", "ppv", "npv", "f1", "mcc"):
            assert key in m


class TestWHOThreshold:
    def test_sensitivity_target_met(self, noisy_scores: tuple) -> None:
        from evaluation.metrics import find_who_threshold  # type: ignore[import-untyped]
        yt, yp = noisy_scores
        result = find_who_threshold(yt, yp, sensitivity_target=0.90)
        assert result["sensitivity"] >= 0.88

    def test_returns_expected_keys(self, noisy_scores: tuple) -> None:
        from evaluation.metrics import find_who_threshold  # type: ignore[import-untyped]
        yt, yp = noisy_scores
        result = find_who_threshold(yt, yp)
        for key in ("threshold", "sensitivity", "specificity", "who_tpp_met"):
            assert key in result


class TestBootstrapCI:
    def test_ci_contains_estimate(self, noisy_scores: tuple) -> None:
        from evaluation.metrics import bootstrap_ci  # type: ignore[import-untyped]
        yt, yp = noisy_scores
        result = bootstrap_ci(yt, yp, roc_auc_score, n_bootstrap=500)
        assert result["ci_lower"] <= result["estimate"] <= result["ci_upper"]

    def test_ci_width_reasonable(self, noisy_scores: tuple) -> None:
        from evaluation.metrics import bootstrap_ci  # type: ignore[import-untyped]
        yt, yp = noisy_scores
        result = bootstrap_ci(yt, yp, roc_auc_score, n_bootstrap=500)
        assert 0.0 < result["ci_upper"] - result["ci_lower"] < 0.3


class TestCalibrationMetrics:
    def test_brier_perfect_is_zero(self, perfect_scores: tuple) -> None:
        from evaluation.metrics import calibration_metrics  # type: ignore[import-untyped]
        yt, yp = perfect_scores
        assert calibration_metrics(yt, yp)["brier"] == pytest.approx(0.0, abs=1e-6)

    def test_ece_in_range(self, noisy_scores: tuple) -> None:
        from evaluation.metrics import calibration_metrics  # type: ignore[import-untyped]
        yt, yp = noisy_scores
        assert 0.0 <= calibration_metrics(yt, yp)["ece"] <= 1.0


class TestPlattCalibrator:
    def test_fit_and_predict(self, noisy_scores: tuple) -> None:
        from evaluation.calibration import PlattCalibrator  # type: ignore[import-untyped]
        yt, yp = noisy_scores
        cal   = PlattCalibrator()
        cal.fit(yp, yt, sensitivity_target=0.90)
        probs = cal.predict_proba(yp)
        assert len(probs) == len(yp)
        assert all(0.0 <= float(p) <= 1.0 for p in probs)

    def test_threshold_achieves_sensitivity(self, noisy_scores: tuple) -> None:
        from evaluation.calibration import PlattCalibrator  # type: ignore[import-untyped]
        from evaluation.metrics import find_who_threshold    # type: ignore[import-untyped]
        yt, yp = noisy_scores
        cal  = PlattCalibrator()
        cal.fit(yp, yt, sensitivity_target=0.90)
        who  = find_who_threshold(yt, cal.predict_proba(yp), sensitivity_target=0.90)
        assert who["sensitivity"] >= 0.88

    def test_save_load_roundtrip(self, noisy_scores: tuple, tmp_path: Path) -> None:
        from evaluation.calibration import PlattCalibrator  # type: ignore[import-untyped]
        yt, yp = noisy_scores
        cal  = PlattCalibrator()
        cal.fit(yp, yt)
        path = tmp_path / "cal.json"
        cal.save(path)
        cal2 = PlattCalibrator.load(path)
        assert cal2.threshold_ == pytest.approx(cal.threshold_, abs=1e-6)
        np.testing.assert_allclose(cal.predict_proba(yp), cal2.predict_proba(yp), rtol=1e-5)


class TestIsotonicCalibrator:
    def test_fit_and_predict(self, noisy_scores: tuple) -> None:
        from evaluation.calibration import IsotonicCalibrator  # type: ignore[import-untyped]
        yt, yp = noisy_scores
        cal   = IsotonicCalibrator()
        cal.fit(yp, yt)
        probs = cal.predict_proba(yp)
        assert len(probs) == len(yp)
