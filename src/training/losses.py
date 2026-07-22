"""
Loss functions for multi-task TB detection training.

L_total = α·FocalLoss(TB classification)
        + β·AsymmetricLoss(multi-label findings)
        + γ·CrossEntropy(active/inactive)

where α=1.0, β=0.3, γ=0.2 per the research plan.

Focal loss (γ=2) is preferred over plain BCE for the heavily imbalanced
TB screening setting (prevalence 1–10 %).  Asymmetric loss is used for the
multi-label findings head to separately tune positive/negative down-weighting.
"""

from __future__ import annotations

from typing import Any

_TORCH_AVAILABLE = False
try:
    import torch as _t      # type: ignore[import-untyped]
    import torch.nn as _nn  # type: ignore[import-untyped]
    _TORCH_AVAILABLE = True
except ImportError:
    _t = None   # type: ignore[assignment]
    _nn = None  # type: ignore[assignment]


def _require(flag: bool, pkg: str) -> None:
    if not flag:
        raise ImportError(f"{pkg} is required. pip install {pkg}")


# ---------------------------------------------------------------------------
# Focal Loss  (binary + multi-class)
# ---------------------------------------------------------------------------

class FocalLoss(object):
    """
    Focal Loss for binary TB classification with label smoothing.

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    Args:
        gamma:          Focusing parameter (default 2.0 per plan).
        alpha:          Class weighting: scalar (applied to positive class) or
                        list [α_neg, α_pos].
        label_smoothing: ε in (0, 1); blends target with uniform distribution.
        reduction:      'mean', 'sum', or 'none'.
    """

    def __new__(
        cls,
        gamma: float = 2.0,
        alpha: float | list[float] = 0.75,
        label_smoothing: float = 0.1,
        reduction: str = "mean",
    ) -> Any:
        _require(_TORCH_AVAILABLE, "torch")
        import torch.nn as nn  # type: ignore[import-untyped]
        import torch  # type: ignore[import-untyped]

        class _FocalLoss(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gamma = gamma
                self.label_smoothing = label_smoothing
                self.reduction = reduction
                if isinstance(alpha, (int, float)):
                    try:
                        self.alpha = torch.tensor([1.0 - float(alpha), float(alpha)])
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"Invalid alpha value {alpha!r}: {exc}") from exc
                else:
                    try:
                        self.alpha = torch.tensor(list(alpha), dtype=torch.float32)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"Invalid alpha list {alpha!r}: {exc}") from exc

            def forward(self, logits: Any, targets: Any) -> Any:
                """
                Args:
                    logits:  (B, 2) raw logits from TB head.
                    targets: (B,) long tensor OR (B, 2) soft float (MixUp/CutMix).
                """
                import torch.nn.functional as F  # type: ignore[import-untyped]

                alpha = self.alpha.to(logits.device)
                n_cls = logits.shape[1]
                eps   = self.label_smoothing

                # Normalise targets → smooth soft distribution (B, n_cls)
                if targets.dim() == 1 or targets.dtype == torch.long:
                    # Hard labels: convert to one-hot first
                    hard = targets.long().view(-1, 1)
                    one_hot = torch.zeros_like(logits).scatter_(1, hard, 1.0)
                    alpha_t = alpha[targets.long()]       # (B,)
                else:
                    # Soft labels already (B, n_cls) float
                    one_hot = targets.float()
                    # alpha_t from the argmax class
                    alpha_t = alpha[(targets.argmax(dim=-1))]  # (B,)

                smooth = one_hot * (1 - eps) + eps / n_cls

                # Focal weights
                log_probs = F.log_softmax(logits, dim=-1)
                probs     = log_probs.exp()
                focal_w   = (1.0 - probs) ** self.gamma  # (B, n_cls)

                loss = -(smooth * focal_w * log_probs).sum(dim=-1)  # (B,)
                loss = loss * alpha_t

                if self.reduction == "mean":
                    return loss.mean()
                if self.reduction == "sum":
                    return loss.sum()
                return loss

        return _FocalLoss()


# ---------------------------------------------------------------------------
# Asymmetric Loss  (multi-label findings head)
# ---------------------------------------------------------------------------

class AsymmetricLoss(object):
    """
    Asymmetric Loss for multi-label TB findings classification.

    Shifts the probability margin for negative samples (γ_neg) higher than
    for positives (γ_pos=0), effectively down-weighting easy negatives.

    Reference: Ben-Baruch et al. (NeurIPS 2021).

    Args:
        gamma_neg: Focusing factor for negative samples (default 4).
        gamma_pos: Focusing factor for positive samples (default 1).
        clip:      Hard probability clip for negatives (default 0.05).
    """

    def __new__(
        cls,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        reduction: str = "mean",
    ) -> Any:
        _require(_TORCH_AVAILABLE, "torch")
        import torch.nn as nn  # type: ignore[import-untyped]

        class _ASL(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gamma_neg  = gamma_neg
                self.gamma_pos  = gamma_pos
                self.clip       = clip
                self.reduction  = reduction

            def forward(self, logits: Any, targets: Any) -> Any:
                """
                Args:
                    logits:  (B, N_findings) raw logits.
                    targets: (B, N_findings) float binary targets.
                """
                import torch  # type: ignore[import-untyped]
                import torch.nn.functional as F  # type: ignore[import-untyped]

                probs = torch.sigmoid(logits)

                # Shift negative probabilities
                if self.clip is not None and self.clip > 0:
                    probs_neg = (probs + self.clip).clamp(max=1.0)
                else:
                    probs_neg = probs

                xs_pos = probs
                xs_neg = 1.0 - probs_neg

                los_pos = targets       * torch.log(xs_pos.clamp(min=1e-8))
                los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=1e-8))

                loss = los_pos + los_neg

                # Asymmetric focusing
                if self.gamma_neg > 0 or self.gamma_pos > 0:
                    pt0   = xs_neg * (1 - targets)
                    pt1   = xs_pos * targets
                    pt    = pt0 + pt1
                    gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
                    loss  = loss * ((1 - pt) ** gamma)

                loss = -loss

                if self.reduction == "mean":
                    return loss.mean()
                if self.reduction == "sum":
                    return loss.sum()
                return loss

        return _ASL()


# ---------------------------------------------------------------------------
# Multi-task combined loss
# ---------------------------------------------------------------------------

class MultiTaskLoss(object):
    """
    Combined multi-task loss:
      L = α·focal(tb) + β·asl(findings) + γ·ce(active/inactive)

    Handles the case where active/inactive labels are missing (-1) by
    masking them out of the CE term.
    """

    def __new__(
        cls,
        cls_weight: float = 1.0,
        findings_weight: float = 0.3,
        active_weight: float = 0.2,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
        label_smoothing: float = 0.1,
        asl_gamma_neg: float = 4.0,
    ) -> Any:
        _require(_TORCH_AVAILABLE, "torch")
        import torch.nn as nn  # type: ignore[import-untyped]
        import torch  # type: ignore[import-untyped]

        class _MTLoss(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.cls_w      = cls_weight
                self.findings_w = findings_weight
                self.active_w   = active_weight
                self.focal      = FocalLoss(focal_gamma, focal_alpha, label_smoothing)
                self.asl        = AsymmetricLoss(asl_gamma_neg)
                self.ce         = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.05)

            def forward(
                self,
                model_out: dict[str, Any],
                tb_labels: Any,
                findings_labels: Any,
                active_labels: Any | None = None,
                lam: float = 1.0,
                labels_b: Any | None = None,
            ) -> dict[str, Any]:
                """
                Args:
                    model_out:       Output dict from TBEnsemble.forward().
                    tb_labels:       (B,) long or (B, 2) soft (for MixUp/CutMix).
                    findings_labels: (B, N_findings) float.
                    active_labels:   (B,) long; -1 = unknown.
                    lam:             MixUp/CutMix lambda (1.0 if no mixing).
                    labels_b:        Second set of labels for MixUp/CutMix.
                """
                tb_logits       = model_out["tb_logits"]
                findings_logits = model_out["findings_logits"]
                active_logits   = model_out.get("active_logits")

                # TB classification loss (with optional MixUp)
                if lam < 1.0 and labels_b is not None:
                    loss_tb = (
                        lam * self.focal(tb_logits, tb_labels)
                        + (1 - lam) * self.focal(tb_logits, labels_b)
                    )
                else:
                    if tb_labels.dim() > 1:
                        tb_labels = tb_labels.argmax(dim=-1)
                    loss_tb = self.focal(tb_logits, tb_labels)

                # Findings loss
                loss_findings = self.asl(findings_logits, findings_labels)

                # Active/inactive loss
                loss_active = torch.tensor(0.0, device=tb_logits.device)
                if active_logits is not None and active_labels is not None:
                    valid = active_labels != -1
                    if valid.any():
                        loss_active = self.ce(
                            active_logits[valid], active_labels[valid]
                        )

                total = (
                    self.cls_w      * loss_tb
                    + self.findings_w * loss_findings
                    + self.active_w   * loss_active
                )

                return {
                    "loss":          total,
                    "loss_tb":       loss_tb.detach(),
                    "loss_findings": loss_findings.detach(),
                    "loss_active":   loss_active.detach(),
                }

        return _MTLoss()
