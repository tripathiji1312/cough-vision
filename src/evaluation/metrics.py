"""
Clinical evaluation metrics for TB screening models.

Primary metric framework follows WHO TPP for triage:
  ≥90% sensitivity at ≥70% specificity.

Computes:
  - AUC-ROC, AUC-PR
  - Sensitivity, specificity, PPV, NPV, F1, MCC at operating thresholds
  - 95% bootstrap confidence intervals
  - Expected Calibration Error (ECE) + Brier score
  - Operating threshold search for WHO sensitivity target
  - Score scaling to 0–100 (CAD4TB convention)
"""

from __future__ import annotations

import math
import warnings
from typing import Any

_NP_AVAILABLE = False
try:
    import numpy as _np  # type: ignore[import-untyped]
    _NP_AVAILABLE = True
except ImportError:
    _np = None  # type: ignore[assignment]

_SK_AVAILABLE = False
try:
    from sklearn import metrics as _sk_metrics  # type: ignore[import-untyped]
    _SK_AVAILABLE = True
except ImportError:
    _sk_metrics = None  # type: ignore[assignment]


def _require(flag: bool, pkg: str) -> None:
    if not flag:
        raise ImportError(f"{pkg} is required. pip install {pkg}")


# ---------------------------------------------------------------------------
# Core metrics at a fixed threshold
# ---------------------------------------------------------------------------

def classification_metrics(
    y_true: Any,
    y_prob: Any,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute sensitivity, specificity, PPV, NPV, F1, and MCC at a threshold.

    Args:
        y_true:    Binary int array (N,) — 0=Normal, 1=TB.
        y_prob:    Float probability array (N,) — P(TB).
        threshold: Decision threshold (default 0.5; calibrated per site).

    Returns:
        Dict of float metrics.
    """
    _require(_NP_AVAILABLE, "numpy")
    _require(_SK_AVAILABLE, "scikit-learn")
    import numpy as np  # type: ignore[import-untyped]
    from sklearn.metrics import (  # type: ignore[import-untyped]
        confusion_matrix, f1_score, matthews_corrcoef,
    )

    try:
        yt = np.asarray(y_true, dtype=int)
        yp = np.asarray(y_prob, dtype=float)
        yhat = (yp >= threshold).astype(int)

        cm = confusion_matrix(yt, yhat, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        sensitivity  = tp / max(tp + fn, 1)
        specificity  = tn / max(tn + fp, 1)
        ppv          = tp / max(tp + fp, 1)
        npv          = tn / max(tn + fn, 1)
        f1           = float(f1_score(yt, yhat, zero_division=0.0))  # type: ignore[call-overload]
        mcc          = float(matthews_corrcoef(yt, yhat)) if len(np.unique(yt)) > 1 else 0.0

        return {
            "threshold":   threshold,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ppv":         ppv,
            "npv":         npv,
            "f1":          f1,
            "mcc":         mcc,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n_pos": int(yt.sum()),
            "n_neg": int((1 - yt).sum()),
        }
    except Exception as exc:
        raise RuntimeError(f"classification_metrics failed: {exc}") from exc


# ---------------------------------------------------------------------------
# AUC metrics
# ---------------------------------------------------------------------------

def auc_metrics(y_true: Any, y_prob: Any) -> dict[str, float]:
    """
    Compute AUC-ROC and AUC-PR.

    AUC-PR is more informative than AUC-ROC for rare diseases (TB prevalence
    1–10 % in screening populations).
    """
    _require(_NP_AVAILABLE, "numpy")
    _require(_SK_AVAILABLE, "scikit-learn")
    import numpy as np  # type: ignore[import-untyped]
    from sklearn.metrics import (  # type: ignore[import-untyped]
        roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    )

    try:
        yt = np.asarray(y_true, dtype=int)
        yp = np.asarray(y_prob, dtype=float)

        if len(np.unique(yt)) < 2:
            warnings.warn("Only one class present — AUC is undefined.", stacklevel=2)
            return {"auc_roc": float("nan"), "auc_pr": float("nan")}

        auc_roc = float(roc_auc_score(yt, yp))
        auc_pr  = float(average_precision_score(yt, yp))

        fpr, tpr, roc_thresholds = roc_curve(yt, yp)
        prec, rec, pr_thresholds = precision_recall_curve(yt, yp)

        return {
            "auc_roc":       auc_roc,
            "auc_pr":        auc_pr,
            "fpr":           fpr.tolist(),
            "tpr":           tpr.tolist(),
            "roc_thresholds": roc_thresholds.tolist(),
            "precision":     prec.tolist(),
            "recall":        rec.tolist(),
        }
    except Exception as exc:
        raise RuntimeError(f"auc_metrics failed: {exc}") from exc


# ---------------------------------------------------------------------------
# WHO TPP operating point search
# ---------------------------------------------------------------------------

def find_who_threshold(
    y_true: Any,
    y_prob: Any,
    sensitivity_target: float = 0.90,
) -> dict[str, float]:
    """
    Find the **highest** decision threshold that still achieves
    ``sensitivity_target``, then report the specificity at that point.

    sklearn's ``roc_curve`` returns thresholds in **descending** order, so
    ``tpr`` increases with index.  The correct operating point is the
    *first* index where ``tpr >= target`` — that is the strictest threshold
    (highest score cut-off) that still meets sensitivity, giving the best
    achievable specificity.

    WHO TPP: ≥90% sensitivity at ≥70% specificity for triage use.

    Returns:
        Dict with keys: threshold, sensitivity, specificity, who_tpp_met.
    """
    _require(_NP_AVAILABLE, "numpy")
    _require(_SK_AVAILABLE, "scikit-learn")
    import numpy as np  # type: ignore[import-untyped]
    from sklearn.metrics import roc_curve  # type: ignore[import-untyped]

    try:
        yt = np.asarray(y_true, dtype=int)
        yp = np.asarray(y_prob, dtype=float)

        fpr, tpr, thresholds = roc_curve(yt, yp)

        # sklearn returns thresholds in descending order, so tpr is non-decreasing.
        # valid_idx[0] is the FIRST (highest-threshold) point where sens ≥ target,
        # giving the best specificity at the required sensitivity level.
        valid_idx = np.where(tpr >= sensitivity_target)[0]
        if len(valid_idx) == 0:
            warnings.warn(
                f"No threshold achieves sensitivity ≥ {sensitivity_target:.0%}.",
                stacklevel=2,
            )
            idx = int(np.argmax(tpr))
        else:
            idx = int(valid_idx[0])  # highest threshold that hits the target

        chosen_threshold  = float(thresholds[idx])
        achieved_sens     = float(tpr[idx])
        achieved_spec     = float(1.0 - fpr[idx])
        who_met           = (
            achieved_sens >= sensitivity_target and achieved_spec >= 0.70
        )

        return {
            "threshold":   chosen_threshold,
            "sensitivity": achieved_sens,
            "specificity": achieved_spec,
            "who_tpp_met": who_met,
        }
    except Exception as exc:
        raise RuntimeError(f"find_who_threshold failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true: Any,
    y_prob: Any,
    metric_fn: Any,
    n_bootstrap: int = 2000,
    ci_alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """
    Compute bootstrap 95% CI for a scalar metric function.

    Args:
        y_true:     Ground truth binary labels.
        y_prob:     Predicted probabilities.
        metric_fn:  Callable(y_true, y_prob) → float.
        n_bootstrap: Number of bootstrap samples.
        ci_alpha:   Significance level (0.05 → 95% CI).
        seed:       RNG seed.

    Returns:
        Dict with keys: estimate, ci_lower, ci_upper.
    """
    _require(_NP_AVAILABLE, "numpy")
    import numpy as np  # type: ignore[import-untyped]

    try:
        rng = np.random.RandomState(seed)
        yt  = np.asarray(y_true, dtype=int)
        yp  = np.asarray(y_prob, dtype=float)
        n   = len(yt)

        estimate = float(metric_fn(yt, yp))
        boot_vals: list[float] = []

        for _ in range(n_bootstrap):
            idx = rng.randint(0, n, size=n)
            yt_b = yt[idx]
            yp_b = yp[idx]
            if len(np.unique(yt_b)) < 2:
                continue
            try:
                boot_vals.append(float(metric_fn(yt_b, yp_b)))
            except Exception:  # noqa: BLE001
                pass

        if not boot_vals:
            return {"estimate": estimate, "ci_lower": float("nan"), "ci_upper": float("nan")}

        boots = np.array(boot_vals)
        lo    = float(np.percentile(boots, 100 * ci_alpha / 2))
        hi    = float(np.percentile(boots, 100 * (1 - ci_alpha / 2)))
        return {"estimate": estimate, "ci_lower": lo, "ci_upper": hi}

    except Exception as exc:
        raise RuntimeError(f"bootstrap_ci failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def calibration_metrics(
    y_true: Any,
    y_prob: Any,
    n_bins: int = 10,
) -> dict[str, Any]:
    """
    Compute Expected Calibration Error (ECE) and Brier score.

    Args:
        y_true: Binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins for ECE.

    Returns:
        Dict with keys: ece, brier_score, bin_accs, bin_confs, bin_counts.
    """
    _require(_NP_AVAILABLE, "numpy")
    import numpy as np  # type: ignore[import-untyped]

    try:
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_prob, dtype=float)
        n  = len(yt)

        # Brier score
        brier = float(np.mean((yp - yt) ** 2))

        # ECE
        bins       = np.linspace(0.0, 1.0, n_bins + 1)
        bin_accs:   list[float] = []
        bin_confs:  list[float] = []
        bin_counts: list[int]   = []
        ece = 0.0

        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (yp >= lo) & (yp < hi)
            if lo == bins[-2]:        # include 1.0 in last bin
                mask |= (yp == 1.0)
            count = int(mask.sum())
            if count == 0:
                bin_accs.append(0.0)
                bin_confs.append(float((lo + hi) / 2))
                bin_counts.append(0)
                continue
            acc  = float(yt[mask].mean())
            conf = float(yp[mask].mean())
            bin_accs.append(acc)
            bin_confs.append(conf)
            bin_counts.append(count)
            ece += (count / n) * abs(acc - conf)

        return {
            "ece":        ece,
            "brier":      brier,
            "bin_accs":   bin_accs,
            "bin_confs":  bin_confs,
            "bin_counts": bin_counts,
        }
    except Exception as exc:
        raise RuntimeError(f"calibration_metrics failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------

def evaluate(
    y_true: Any,
    y_prob: Any,
    sensitivity_target: float = 0.90,
    n_bootstrap: int = 2000,
    site: str = "global",
) -> dict[str, Any]:
    """
    Run the complete evaluation suite and return a structured report.

    Args:
        y_true:             Binary labels (N,).
        y_prob:             TB probabilities (N,).
        sensitivity_target: WHO TPP sensitivity floor (0.90).
        n_bootstrap:        Bootstrap samples for CIs.
        site:               Site identifier (for multi-center reports).

    Returns:
        Nested dict with auc, who_operating_point, metrics, calibration,
        and bootstrap_ci sub-dicts.
    """
    _require(_NP_AVAILABLE, "numpy")
    import numpy as np  # type: ignore[import-untyped]
    from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]

    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=float)

    report: dict[str, Any] = {"site": site, "n_samples": len(yt)}

    # AUC
    try:
        report["auc"] = auc_metrics(yt, yp)
    except Exception as exc:  # noqa: BLE001
        report["auc"] = {"error": str(exc)}

    # WHO operating point
    try:
        who = find_who_threshold(yt, yp, sensitivity_target)
        report["who_operating_point"] = who
        # Full metrics at the WHO threshold
        report["metrics_at_who_threshold"] = classification_metrics(
            yt, yp, threshold=who["threshold"]
        )
    except Exception as exc:  # noqa: BLE001
        report["who_operating_point"] = {"error": str(exc)}

    # Calibration
    try:
        report["calibration"] = calibration_metrics(yt, yp)
    except Exception as exc:  # noqa: BLE001
        report["calibration"] = {"error": str(exc)}

    # Bootstrap CI on AUC-ROC
    if len(np.unique(yt)) > 1:
        try:
            report["auc_roc_ci"] = bootstrap_ci(yt, yp, roc_auc_score, n_bootstrap)
        except Exception as exc:  # noqa: BLE001
            report["auc_roc_ci"] = {"error": str(exc)}

    return report
