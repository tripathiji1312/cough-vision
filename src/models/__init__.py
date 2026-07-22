"""
TBEnsemble — the complete hybrid CNN+ViT TB detection model.

Wires together:
  - CNN branch (DenseNet-121 or EfficientNet-B4)
  - ViT branch (Conv-stem ViT-S/16)
  - Attention-gated late fusion
  - Three multi-task heads (TB classification, findings, active/inactive)
  - Grad-CAM hook registration on the CNN branch

One import gives you the full inference pipeline::

    from models import TBEnsemble, build_ensemble
    model = build_ensemble(cfg)
    out   = model(image_tensor)
    # out['tb_prob']         → (B,)   probability of TB (0–1)
    # out['tb_score']        → (B,)   CAD4TB-style 0–100 score
    # out['findings_logits'] → (B, 6) multi-label finding logits
    # out['gradcam']         → (B, H, W) heatmaps for positive cases
"""

from __future__ import annotations

import warnings
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


def build_ensemble(cfg: Any | None = None) -> Any:
    """
    Build the full TBEnsemble model from a :class:`Config` object.

    Falls back to sensible defaults when ``cfg`` is None.

    Returns an ``nn.Module`` with::

        forward(
            x_cnn,                     # (B, 3, 224, 224)
            x_vit=None,                # (B, 3, 384, 384) — same as x_cnn if None
            return_gradcam=False,      # compute Grad-CAM on positive cases
            lung_mask=None,            # (B, 1, H, W) binary mask for CAM masking
        ) → dict[str, Tensor]
    """
    _require(_TORCH_AVAILABLE, "torch")
    import torch.nn as nn  # type: ignore[import-untyped]
    import torch  # type: ignore[import-untyped]

    # --- resolve config fields with defaults --------------------------------
    cnn_name       = "densenet121"
    cnn_pretrained = "imagenet"
    cnn_ckpt       = None
    cnn_drop       = 0.2
    vit_name       = "vit_small_patch16_384"
    vit_pretrained = "imagenet"
    vit_ckpt       = None
    vit_stem_ch    = (32, 64, 128)
    vit_drop       = 0.1
    fusion_method  = "attention"
    fusion_hidden  = 512
    fusion_drop    = 0.3
    n_findings     = 6
    include_active = True
    heads_drop     = 0.3

    if cfg is not None:
        try:
            cnn_name       = cfg.cnn.name
            cnn_pretrained = cfg.cnn.pretrained
            cnn_ckpt       = cfg.cnn.pretrained_ckpt
            cnn_drop       = cfg.cnn.drop_rate
            vit_name       = cfg.vit.name
            vit_pretrained = cfg.vit.pretrained
            vit_ckpt       = cfg.vit.pretrained_ckpt
            vit_stem_ch    = cfg.vit.conv_stem_channels
            vit_drop       = cfg.vit.drop_rate
            fusion_method  = cfg.fusion.method
            fusion_hidden  = cfg.fusion.hidden_dim
            fusion_drop    = cfg.fusion.dropout
            n_findings     = cfg.heads.findings_classes
            include_active = cfg.heads.active_inactive
            heads_drop     = 0.3
        except AttributeError as exc:
            warnings.warn(f"Config parsing partial: {exc}. Using defaults.", stacklevel=2)

    from .backbones.cnn import build_cnn_backbone, _CNN_FEATURE_DIMS  # type: ignore[import-untyped]
    from .backbones.vit import build_vit_backbone   # type: ignore[import-untyped]
    from .backbones.fusion import build_fusion_module  # type: ignore[import-untyped]
    from .heads import MultiTaskHead, scale_to_cad4tb  # type: ignore[import-untyped]
    from .interpretability import GradCAM, get_target_layer  # type: ignore[import-untyped]

    cnn_branch  = build_cnn_backbone(cnn_name, cnn_pretrained, cnn_ckpt, cnn_drop)
    vit_branch  = build_vit_backbone(vit_name, vit_pretrained, vit_ckpt,
                                     use_conv_stem=True, stem_channels=vit_stem_ch,
                                     drop_rate=vit_drop)
    cnn_dim: int = _CNN_FEATURE_DIMS.get(cnn_name, 1024)
    vit_dim = vit_branch.feature_dim

    fusion_mod  = build_fusion_module(cnn_dim, vit_dim, fusion_hidden, fusion_drop,
                                      fusion_method)
    heads: Any  = MultiTaskHead(fusion_hidden, n_findings, include_active, heads_drop)

    class _TBEnsemble(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cnn    = cnn_branch
            self.vit    = vit_branch
            self.fusion = fusion_mod
            self.heads  = heads
            self._gradcam: GradCAM | None = None

        # ------------------------------------------------------------------
        # Grad-CAM
        # ------------------------------------------------------------------

        def enable_gradcam(self) -> None:
            """
            Attach Grad-CAM hooks to the CNN branch's last conv layer,
            but score via the FULL ensemble forward pass so that
            tb_logits (not raw CNN features) drives the gradient.
            """
            layer = get_target_layer(self.cnn, cnn_name)
            # Pass `self` (the full ensemble) so the CAM forward call
            # produces a dict with 'tb_logits' rather than a raw vector.
            self._gradcam = GradCAM(self, layer)

        def disable_gradcam(self) -> None:
            if self._gradcam is not None:
                self._gradcam.remove_hooks()
                self._gradcam = None

        # ------------------------------------------------------------------
        # Forward
        # ------------------------------------------------------------------

        def forward(
            self,
            x_cnn: Any,
            x_vit: Any | None = None,
            return_gradcam: bool = False,
            lung_mask: Any | None = None,
        ) -> dict[str, Any]:
            """
            Args:
                x_cnn:          (B, 3, 224, 224) — CNN branch input.
                x_vit:          (B, 3, 384, 384) — ViT branch input.
                                If None, x_cnn is resized on-the-fly.
                return_gradcam: Compute Grad-CAM heatmaps.
                lung_mask:      (B, 1, H, W) binary lung mask for CAM masking.

            Returns dict with keys:
                tb_logits, tb_prob, tb_score (0-100),
                findings_logits, active_logits (or None),
                gradcam (or None).
            """
            import torch.nn.functional as F  # type: ignore[import-untyped]

            # ViT input: use same image resized if not provided separately
            if x_vit is None:
                try:
                    vit_size = self.vit.vit.patch_embed.img_size
                    if hasattr(vit_size, "__iter__"):
                        vit_h, vit_w = int(vit_size[0]), int(vit_size[1])
                    else:
                        vit_h = vit_w = int(vit_size)
                except (AttributeError, TypeError):
                    vit_h = vit_w = 384
                x_vit = F.interpolate(x_cnn, size=(vit_h, vit_w),
                                      mode="bilinear", align_corners=False)

            cnn_feat = self.cnn(x_cnn)          # (B, cnn_dim)
            vit_feat = self.vit(x_vit)          # (B, vit_dim)
            fused    = self.fusion(cnn_feat, vit_feat)  # (B, hidden_dim)

            out = self.heads(fused)
            out["tb_score"] = scale_to_cad4tb(out["tb_prob"])

            # Grad-CAM on positive cases
            if return_gradcam and self._gradcam is not None:
                try:
                    B = x_cnn.shape[0]
                    cams = []
                    for i in range(B):
                        cam_i = self._gradcam(x_cnn[i:i+1], class_idx=1)
                        cams.append(cam_i)
                    import numpy as np  # type: ignore[import-untyped]
                    out["gradcam"] = np.stack(cams, axis=0)  # (B, H', W')
                except Exception as exc:  # noqa: BLE001
                    warnings.warn(f"Grad-CAM failed: {exc}", stacklevel=2)
                    out["gradcam"] = None
            else:
                out["gradcam"] = None

            return out

        # ------------------------------------------------------------------
        # Staged fine-tuning helpers
        # ------------------------------------------------------------------

        def freeze_backbones(self) -> None:
            """Stage 2a: freeze both backbones, train heads only."""
            from .backbones.cnn import freeze_backbone  # type: ignore[import-untyped]
            from .backbones.vit import freeze_vit       # type: ignore[import-untyped]
            freeze_backbone(self.cnn)
            freeze_vit(self.vit)

        def unfreeze_backbones(self) -> None:
            """Stage 2b: full fine-tune."""
            from .backbones.cnn import unfreeze_backbone  # type: ignore[import-untyped]
            from .backbones.vit import unfreeze_vit       # type: ignore[import-untyped]
            unfreeze_backbone(self.cnn)
            unfreeze_vit(self.vit)

        def get_parameter_groups(
            self,
            backbone_lr: float = 1e-5,
            head_lr: float = 1e-3,
        ) -> list[dict[str, Any]]:
            """
            Discriminative learning rates for AdamW.

            Backbone (CNN + ViT) parameters use backbone_lr to preserve
            pretrained representations.  Everything else (fusion + heads)
            uses head_lr.  Uses named-submodule traversal rather than
            id()-set membership so it remains correct under any future
            parameter sharing (audit DEBT-04).
            """
            backbone_params: list[Any] = []
            head_params: list[Any]     = []
            backbone_names = {"cnn", "vit"}

                for name, sub in self.named_children():
                    params = list(sub.parameters())
                    if name in backbone_names:
                        backbone_params.extend(params)
                    else:
                        head_params.extend(params)
            return [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": head_params,     "lr": head_lr},
            ]

    return _TBEnsemble()
