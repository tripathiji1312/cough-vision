"""
Multi-task output heads for the TB detection ensemble.

Head 1 — TB binary classification  (primary, drives clinical score)
Head 2 — Multi-label TB findings   (auxiliary: cavitation, consolidation,
          pleural effusion, hilar LAD, fibrosis, nodules — improves CAM quality)
Head 3 — Active vs inactive TB     (optional, requires active/inactive labels)

The ``MultiTaskHead`` module wraps all three heads and exposes a single
``.forward()`` that returns a named dict of logits.
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


N_FINDINGS = 6
FINDINGS_NAMES = [
    "cavitation",
    "consolidation",
    "pleural_effusion",
    "hilar_lad",
    "fibrosis",
    "nodule",
]


class MultiTaskHead(object):
    """
    Factory that returns an ``nn.Module`` with three heads.

    Args:
        input_dim:       Fused feature dimension (from fusion module).
        n_findings:      Number of TB-relevant finding classes (default 6).
        include_active:  Build the active/inactive head (Head 3).
        dropout:         Dropout before each head.

    Returns:
        nn.Module whose ``.forward(x)`` returns::

            {
              'tb_logits':          (B, 2),   # raw logits
              'tb_prob':            (B,),     # sigmoid probability of TB
              'findings_logits':    (B, N_FINDINGS),
              'active_logits':      (B, 2) or None,
            }
    """

    def __new__(
        cls,
        input_dim: int = 512,
        n_findings: int = N_FINDINGS,
        include_active: bool = True,
        dropout: float = 0.3,
    ) -> Any:
        _require(_TORCH_AVAILABLE, "torch")
        import torch.nn as nn  # type: ignore[import-untyped]
        import torch  # type: ignore[import-untyped]

        class _MultiTaskHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_findings = n_findings
                self._include_active = include_active

                # Head 1: TB binary (2-class → softmax, or scalar → sigmoid)
                self.tb_head = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(input_dim, 256),
                    nn.LayerNorm(256),
                    nn.GELU(),
                    nn.Dropout(dropout / 2),
                    nn.Linear(256, 2),
                )

                # Head 2: Multi-label TB findings
                self.findings_head = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(input_dim, 256),
                    nn.GELU(),
                    nn.Linear(256, n_findings),
                )

                # Head 3: Active vs inactive (optional)
                if include_active:
                    self.active_head: nn.Module | None = nn.Sequential(
                        nn.Dropout(dropout),
                        nn.Linear(input_dim, 128),
                        nn.GELU(),
                        nn.Linear(128, 2),
                    )
                else:
                    self.active_head = None

            def forward(self, x: Any) -> dict[str, Any]:
                tb_logits = self.tb_head(x)               # (B, 2)
                # Calibrated probability of the positive (TB) class
                tb_prob = torch.softmax(tb_logits, dim=-1)[:, 1]  # (B,)
                findings_logits = self.findings_head(x)   # (B, N_FINDINGS)
                active_logits = (
                    self.active_head(x) if self.active_head is not None else None
                )
                return {
                    "tb_logits":       tb_logits,
                    "tb_prob":         tb_prob,
                    "findings_logits": findings_logits,
                    "active_logits":   active_logits,
                }

        return _MultiTaskHead()


# ---------------------------------------------------------------------------
# Score scaling: raw probability → 0-100 (CAD4TB-compatible)
# ---------------------------------------------------------------------------

def scale_to_cad4tb(prob: Any) -> Any:
    """
    Convert a scalar TB probability in [0, 1] to a 0–100 integer score
    matching the CAD4TB clinical display convention.

    Works on Python floats, numpy scalars/arrays, or torch tensors.
    """
    try:
        score = prob * 100.0
        # Keep as same type (tensor or ndarray) for downstream use
        return score
    except Exception as exc:
        raise TypeError(f"Cannot scale prob of type {type(prob)}: {exc}") from exc
