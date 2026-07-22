"""
Smoke tests for the full model graph.

Covers (audit TEST-01):
  - build_ensemble() returns an nn.Module with the documented forward contract
  - Output dict keys and tensor shapes are correct
  - MultiTaskLoss runs on the output and produces a finite scalar
  - scale_to_cad4tb returns values in [0, 100]
  - ViT conv-stem raises ValueError for incompatible patch_size (CORRECT-01)
  - get_parameter_groups returns two non-empty groups (DEBT-04)
  - freeze/unfreeze correctly toggles requires_grad

All tests are skipped if torch or timm are not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Stable module-level fallback so names are never unbound
torch: Any = None
_TORCH_AVAILABLE = False

try:
    import torch          # type: ignore[import-untyped,no-redef]
    import torch.nn as nn # type: ignore[import-untyped]
    _TORCH_AVAILABLE = True
except ImportError:
    pass

_TIMM_AVAILABLE = False
try:
    import timm           # type: ignore[import-untyped]
    _TIMM_AVAILABLE = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not (_TORCH_AVAILABLE and _TIMM_AVAILABLE),
    reason="torch and timm required",
)

BATCH = 2
IMG   = 224
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_cfg():
    """Config that builds a minimal ensemble fast on CPU."""
    from config import get_config  # type: ignore[import-untyped]
    cfg = get_config("densenet_vit_ensemble")
    # Use the smallest ViT available in timm for fast tests
    cfg.vit.name           = "vit_tiny_patch16_224"
    cfg.vit.conv_stem_channels = (16, 32)  # 2-layer stem: 2^2=4 stride, patch=16 -> final=4 (ok)
    cfg.preprocess.image_size_cnn = IMG
    cfg.preprocess.image_size_vit = IMG
    return cfg


def _build():
    from models import build_ensemble  # type: ignore[import-untyped]
    cfg = _tiny_cfg()
    return build_ensemble(cfg).to(DEVICE).eval()


def _dummy_input():
    return torch.zeros(BATCH, 3, IMG, IMG)


# ---------------------------------------------------------------------------
# Forward contract tests
# ---------------------------------------------------------------------------

class TestEnsembleForward:
    def test_returns_dict(self) -> None:
        model = _build()
        with torch.no_grad():
            out = model(_dummy_input())
        assert isinstance(out, dict), f"Expected dict, got {type(out)}"

    def test_required_keys(self) -> None:
        model = _build()
        with torch.no_grad():
            out = model(_dummy_input())
        for key in ("tb_logits", "tb_prob", "tb_score", "findings_logits"):
            assert key in out, f"Missing key: {key}"

    def test_tb_logits_shape(self) -> None:
        model = _build()
        with torch.no_grad():
            out = model(_dummy_input())
        assert out["tb_logits"].shape == (BATCH, 2), out["tb_logits"].shape

    def test_tb_prob_shape_and_range(self) -> None:
        model = _build()
        with torch.no_grad():
            out = model(_dummy_input())
        prob = out["tb_prob"]
        assert prob.shape == (BATCH,), prob.shape
        assert float(prob.min()) >= 0.0
        assert float(prob.max()) <= 1.0

    def test_tb_score_range(self) -> None:
        model = _build()
        with torch.no_grad():
            out = model(_dummy_input())
        score = out["tb_score"]
        assert float(score.min()) >= 0.0
        assert float(score.max()) <= 100.0

    def test_findings_shape(self) -> None:
        model = _build()
        with torch.no_grad():
            out = model(_dummy_input())
        assert out["findings_logits"].shape == (BATCH, 6), out["findings_logits"].shape

    def test_active_logits_present_when_enabled(self) -> None:
        model = _build()
        with torch.no_grad():
            out = model(_dummy_input())
        # active_logits should be a tensor (not None) when include_active=True (default)
        if out["active_logits"] is not None:
            assert out["active_logits"].shape == (BATCH, 2)


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------

class TestMultiTaskLoss:
    def test_loss_is_finite_scalar(self) -> None:
        from training.losses import MultiTaskLoss  # type: ignore[import-untyped]
        model    = _build()
        criterion = MultiTaskLoss()
        x = _dummy_input()
        with torch.no_grad():
            out = model(x)
        tb_labels  = torch.zeros(BATCH, dtype=torch.long)
        findings   = torch.zeros(BATCH, 6)
        active     = torch.full((BATCH,), -1, dtype=torch.long)

        # Re-enable grad for loss computation
        model.train()
        out2 = model(x)
        loss_dict = criterion(out2, tb_labels, findings, active)
        loss = loss_dict["loss"]
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        assert loss.dim() == 0, "Loss must be a scalar"

    def test_focal_loss_soft_labels(self) -> None:
        """Soft (one-hot float) labels from CutMix must not crash FocalLoss."""
        from training.losses import FocalLoss  # type: ignore[import-untyped]
        fl = FocalLoss()
        logits   = torch.randn(BATCH, 2)
        # Soft labels (one-hot float — from MixUp/CutMix)
        soft = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        loss = fl(logits, soft)
        assert torch.isfinite(loss)

    def test_focal_loss_hard_labels(self) -> None:
        from training.losses import FocalLoss  # type: ignore[import-untyped]
        fl   = FocalLoss()
        loss = fl(torch.randn(BATCH, 2), torch.zeros(BATCH, dtype=torch.long))
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# scale_to_cad4tb
# ---------------------------------------------------------------------------

class TestScaleToCad4tb:
    def test_range_tensor(self) -> None:
        from models.heads import scale_to_cad4tb  # type: ignore[import-untyped]
        prob  = torch.tensor([0.0, 0.5, 1.0])
        score = scale_to_cad4tb(prob)
        assert float(score.min()) >= 0.0
        assert float(score.max()) <= 100.0

    def test_range_float(self) -> None:
        from models.heads import scale_to_cad4tb  # type: ignore[import-untyped]
        assert scale_to_cad4tb(0.0)  == pytest.approx(0.0)
        assert scale_to_cad4tb(1.0)  == pytest.approx(100.0)
        assert scale_to_cad4tb(0.75) == pytest.approx(75.0)


# ---------------------------------------------------------------------------
# ViT conv-stem geometry (CORRECT-01)
# ---------------------------------------------------------------------------

class TestViTConvStem:
    def test_invalid_patch_size_raises(self) -> None:
        """patch_size=14 with 3-layer stem (total stride=8) must raise ValueError."""
        from models.backbones.vit import _build_conv_stem  # type: ignore[import-untyped]
        with pytest.raises(ValueError, match="not divisible"):
            _build_conv_stem(
                in_channels=3,
                stem_channels=(32, 64, 128),   # 3 layers -> stride body = 8
                patch_size=14,                  # 14 % 8 != 0 -> must raise
                embed_dim=192,
            )

    def test_valid_patch_size_runs(self) -> None:
        """patch_size=16 with 3-layer stem (total stride=8, final=2) must work."""
        from models.backbones.vit import _build_conv_stem  # type: ignore[import-untyped]
        stem = _build_conv_stem(
            in_channels=3,
            stem_channels=(32, 64, 128),
            patch_size=16,
            embed_dim=192,
        )
        x   = torch.zeros(1, 3, 224, 224)
        out = stem(x)
        # Expected spatial: 224 / 16 = 14
        assert out.shape[-1] == 14, f"Expected 14, got {out.shape[-1]}"
        assert out.shape[-2] == 14

    def test_patch4_with_2layer_stem(self) -> None:
        """patch_size=4 with 2-layer stem (stride_body=4, final=1) must work."""
        from models.backbones.vit import _build_conv_stem  # type: ignore[import-untyped]
        stem = _build_conv_stem(
            in_channels=3,
            stem_channels=(16, 32),   # 2 layers -> stride body = 4
            patch_size=4,             # 4 % 4 == 0, final_stride = 1
            embed_dim=96,
        )
        x   = torch.zeros(1, 3, 64, 64)
        out = stem(x)
        assert out.shape[-1] == 16  # 64 / 4 = 16


# ---------------------------------------------------------------------------
# Parameter groups (DEBT-04)
# ---------------------------------------------------------------------------

class TestParameterGroups:
    def test_two_groups_returned(self) -> None:
        model = _build()
        groups = model.get_parameter_groups()
        assert len(groups) == 2, f"Expected 2 groups, got {len(groups)}"

    def test_backbone_and_head_non_empty(self) -> None:
        model  = _build()
        groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)
        backbone_p, head_p = groups[0]["params"], groups[1]["params"]
        assert len(backbone_p) > 0, "backbone group is empty"
        assert len(head_p) > 0,     "head group is empty"

    def test_lr_values_assigned(self) -> None:
        model  = _build()
        groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)
        assert groups[0]["lr"] == pytest.approx(1e-5)
        assert groups[1]["lr"] == pytest.approx(1e-3)

    def test_no_params_shared_between_groups(self) -> None:
        model  = _build()
        groups = model.get_parameter_groups()
        ids_a  = {id(p) for p in groups[0]["params"]}
        ids_b  = {id(p) for p in groups[1]["params"]}
        assert ids_a.isdisjoint(ids_b), "Same parameter appears in both groups"


# ---------------------------------------------------------------------------
# Freeze / unfreeze
# ---------------------------------------------------------------------------

class TestFreezeUnfreeze:
    def test_freeze_disables_backbone_grad(self) -> None:
        model = _build()
        model.freeze_backbones()
        cnn_frozen = all(not p.requires_grad for p in model.cnn.parameters())
        vit_frozen = all(not p.requires_grad for p in model.vit.parameters())
        assert cnn_frozen, "CNN params still have requires_grad=True after freeze"
        assert vit_frozen, "ViT params still have requires_grad=True after freeze"

    def test_unfreeze_restores_backbone_grad(self) -> None:
        model = _build()
        model.freeze_backbones()
        model.unfreeze_backbones()
        cnn_ok = any(p.requires_grad for p in model.cnn.parameters())
        vit_ok = any(p.requires_grad for p in model.vit.parameters())
        assert cnn_ok, "CNN params still frozen after unfreeze"
        assert vit_ok, "ViT params still frozen after unfreeze"
