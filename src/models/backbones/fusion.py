"""
Late attention-gated feature fusion of CNN + ViT branches.

Three fusion methods are supported:
  1. ``'concat'``          — plain concatenation + MLP
  2. ``'attention'``       — 1-layer attention gating (recommended)
  3. ``'deep_layer_agg'``  — deep-layer aggregation inspired by Nature 2026 SOTA
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


def build_fusion_module(
    cnn_dim: int = 1024,
    vit_dim: int = 384,
    hidden_dim: int = 512,
    dropout: float = 0.3,
    method: str = "attention",
) -> Any:
    """
    Build a feature fusion module that takes two (B, D) feature vectors
    and returns a single (B, hidden_dim) fused representation.

    Args:
        cnn_dim:    CNN branch feature dimension.
        vit_dim:    ViT branch feature dimension.
        hidden_dim: Output dimension of the fused representation.
        dropout:    Dropout applied to fused features.
        method:     Fusion strategy: 'concat', 'attention', 'deep_layer_agg'.

    Returns:
        nn.Module with ``.forward(cnn_feat, vit_feat)`` → (B, hidden_dim).
        ``.output_dim`` attribute set to hidden_dim.
    """
    _require(_TORCH_AVAILABLE, "torch")
    import torch.nn as nn  # type: ignore[import-untyped]
    import torch  # type: ignore[import-untyped]

    if method == "concat":
        class _ConcatFusion(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.output_dim = hidden_dim
                self.proj = nn.Sequential(
                    nn.Linear(cnn_dim + vit_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )

            def forward(self, cnn_feat: Any, vit_feat: Any) -> Any:
                return self.proj(torch.cat([cnn_feat, vit_feat], dim=-1))

        return _ConcatFusion()

    elif method == "attention":
        class _AttentionFusion(nn.Module):
            """
            Attention gating: learn a scalar gate α per branch such that
            output = α·proj(CNN) + (1-α)·proj(ViT), then project to hidden_dim.
            """
            def __init__(self) -> None:
                super().__init__()
                self.output_dim = hidden_dim
                self.proj_cnn = nn.Linear(cnn_dim, hidden_dim)
                self.proj_vit = nn.Linear(vit_dim, hidden_dim)
                # Gate network: [cnn, vit] → 2 scalars (softmax)
                self.gate = nn.Sequential(
                    nn.Linear(cnn_dim + vit_dim, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 2),
                    nn.Softmax(dim=-1),
                )
                self.norm = nn.LayerNorm(hidden_dim)
                self.dropout = nn.Dropout(dropout)

            def forward(self, cnn_feat: Any, vit_feat: Any) -> Any:
                g = self.gate(torch.cat([cnn_feat, vit_feat], dim=-1))  # (B, 2)
                fused = (
                    g[:, 0:1] * self.proj_cnn(cnn_feat)
                    + g[:, 1:2] * self.proj_vit(vit_feat)
                )
                return self.dropout(self.norm(fused))

        return _AttentionFusion()

    elif method == "deep_layer_agg":
        class _DLAFusion(nn.Module):
            """
            Deep-layer aggregation: project each branch independently,
            add, then pass through two MLP layers with residual connection.
            """
            def __init__(self) -> None:
                super().__init__()
                self.output_dim = hidden_dim
                self.proj_cnn = nn.Linear(cnn_dim, hidden_dim)
                self.proj_vit = nn.Linear(vit_dim, hidden_dim)
                self.mlp = nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Dropout(dropout),
                )
                self.norm = nn.LayerNorm(hidden_dim)

            def forward(self, cnn_feat: Any, vit_feat: Any) -> Any:
                x = self.proj_cnn(cnn_feat) + self.proj_vit(vit_feat)
                return self.norm(x + self.mlp(x))  # residual

        return _DLAFusion()

    else:
        raise ValueError(f"Unknown fusion method '{method}'. "
                         "Choose from: 'concat', 'attention', 'deep_layer_agg'.")
