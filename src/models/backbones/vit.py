"""
Vision Transformer backbone branch with Conv2D stem.

Replacing the linear patch-embed projection with 3–4 strided Conv2D layers
is the key modification that makes ViTs work on medical images: the stem
preserves local texture (cavitation walls, nodule borders) that pure
patch-embed discards.

Supported base models (resolved via timm):
  - vit_small_patch16_384   (ViT-S/16 @ 384 px)  — recommended
  - vit_tiny_patch16_224    (ViT-Ti/16 @ 224 px)  — edge/fast
  - swin_tiny_patch4_window7_224                  — Swin-Tiny alternative
  - swin_small_patch4_window7_224                 — Swin-Small

Initialization options: 'imagenet', 'ssl_cxr' (load from checkpoint), 'none'.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

_TORCH_AVAILABLE = False
try:
    import torch as _t      # type: ignore[import-untyped]
    import torch.nn as _nn  # type: ignore[import-untyped]
    _TORCH_AVAILABLE = True
except ImportError:
    _t = None   # type: ignore[assignment]
    _nn = None  # type: ignore[assignment]

_TIMM_AVAILABLE = False
try:
    import timm as _timm    # type: ignore[import-untyped]
    _TIMM_AVAILABLE = True
except ImportError:
    _timm = None  # type: ignore[assignment]


def _require(flag: bool, pkg: str) -> None:
    if not flag:
        raise ImportError(f"{pkg} is required. pip install {pkg}")


# ---------------------------------------------------------------------------
# Conv2D stem (replaces patch-embed linear projection)
# ---------------------------------------------------------------------------

def _build_conv_stem(
    in_channels: int,
    stem_channels: tuple[int, ...],
    patch_size: int,
    embed_dim: int,
) -> Any:
    """
    Build a multi-layer Conv2D stem that maps (B, C, H, W) →
    (B, embed_dim, H/patch_size, W/patch_size).

    The final layer uses stride=patch_size to produce the same spatial
    layout as a standard ViT patch embed, so the positional embeddings
    remain compatible.

    Args:
        in_channels:   Input channels (3 for RGB CXR).
        stem_channels: Intermediate channel counts (e.g. (32, 64, 128)).
        patch_size:    ViT patch size (e.g. 16) — used for the final stride.
        embed_dim:     ViT embedding dimension (e.g. 384 for ViT-S).
    """
    _require(_TORCH_AVAILABLE, "torch")
    import torch.nn as nn  # type: ignore[import-untyped]

    layers: list[Any] = []
    c_in = in_channels
    for i, c_out in enumerate(stem_channels):
        layers += [
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
        ]
        c_in = c_out

    # Final projection to embed_dim — stride chosen so total downsampling = patch_size
    # e.g. 3-layer stem with stride-2 each = 8× downsampling → need 2× more → stride=2
    remaining_stride = max(1, patch_size // (2 ** len(stem_channels)))
    layers += [
        nn.Conv2d(c_in, embed_dim, kernel_size=remaining_stride + 1,
                  stride=remaining_stride, padding=remaining_stride // 2, bias=False),
        nn.BatchNorm2d(embed_dim),
    ]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# ViT backbone builder
# ---------------------------------------------------------------------------

def build_vit_backbone(
    name: str = "vit_small_patch16_384",
    pretrained: str = "imagenet",
    pretrained_ckpt: str | None = None,
    use_conv_stem: bool = True,
    stem_channels: tuple[int, ...] = (32, 64, 128),
    drop_rate: float = 0.1,
    attn_drop_rate: float = 0.0,
    in_channels: int = 3,
) -> Any:
    """
    Build a ViT feature extractor with optional Conv2D stem.

    Args:
        name:            timm model name.
        pretrained:      ``'imagenet'``, ``'ssl_cxr'``, or ``'none'``.
        pretrained_ckpt: Path for ``'ssl_cxr'`` initialization.
        use_conv_stem:   Replace patch-embed with Conv2D stem.
        stem_channels:   Intermediate channels in the Conv stem.
        drop_rate:       Stochastic depth / MLP dropout.
        attn_drop_rate:  Attention dropout.
        in_channels:     Input channels.

    Returns:
        nn.Module with ``.forward(x)`` → (B, feature_dim) CLS token.
        ``model.feature_dim`` is set accordingly.
    """
    _require(_TORCH_AVAILABLE, "torch")
    _require(_TIMM_AVAILABLE, "timm")
    import torch.nn as nn  # type: ignore[import-untyped]
    import timm  # type: ignore[import-untyped]

    use_imagenet_weights = pretrained == "imagenet"

    class _ViTBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Build timm ViT — num_classes=0 removes head, returns CLS token
            self.vit = timm.create_model(
                name,
                pretrained=use_imagenet_weights,
                num_classes=0,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                in_chans=in_channels,
            )
            self.feature_dim: int = self.vit.num_features

            # Replace patch embedding with Conv2D stem
            if use_conv_stem:
                patch_size = getattr(self.vit, "patch_embed", None)
                p_size = 16  # default
                if patch_size is not None:
                    p_size = getattr(patch_size, "patch_size", 16)
                    # patch_size may be an int or a tuple
                    if not isinstance(p_size, int):
                        try:
                            p_size = int(p_size[0])
                        except (TypeError, IndexError):
                            p_size = 16
                    else:
                        try:
                            p_size = int(p_size)
                        except (TypeError, ValueError):
                            p_size = 16

                self.stem = _build_conv_stem(
                    in_channels=in_channels,
                    stem_channels=stem_channels,
                    patch_size=p_size,
                    embed_dim=self.vit.num_features,
                )
                # Replace patch_embed.proj with identity; inject our stem output
                self._use_stem = True
            else:
                self._use_stem = False

        def forward(self, x: Any) -> Any:
            if self._use_stem:
                # Run Conv stem → flatten to patch sequence → feed into ViT transformer
                import torch  # type: ignore[import-untyped]
                B = x.shape[0]
                feat_map = self.stem(x)                    # (B, D, H', W')
                H_p, W_p = feat_map.shape[-2], feat_map.shape[-1]
                patches = feat_map.flatten(2).transpose(1, 2)  # (B, N, D)

                # Inject patches by bypassing patch_embed
                # Access ViT internals via timm's forward_features
                vit = self.vit
                if hasattr(vit, "cls_token"):
                    cls = vit.cls_token.expand(B, -1, -1)
                    patches = torch.cat([cls, patches], dim=1)
                if hasattr(vit, "pos_embed"):
                    # Interpolate positional embedding if spatial size differs
                    pos = vit.pos_embed
                    if pos.shape[1] != patches.shape[1]:
                        pos = self._interpolate_pos_embed(pos, H_p, W_p)
                    patches = patches + pos
                if hasattr(vit, "pos_drop"):
                    patches = vit.pos_drop(patches)
                if hasattr(vit, "blocks"):
                    for blk in vit.blocks:
                        patches = blk(patches)
                if hasattr(vit, "norm"):
                    patches = vit.norm(patches)
                # Return CLS token
                return patches[:, 0]
            else:
                return self.vit(x)

        @staticmethod
        def _interpolate_pos_embed(pos_embed: Any, h: int, w: int) -> Any:
            """Bilinearly interpolate position embeddings to a new grid size."""
            import torch  # type: ignore[import-untyped]
            import torch.nn.functional as F  # type: ignore[import-untyped]

            cls_pos = pos_embed[:, :1, :]          # (1, 1, D)
            patch_pos = pos_embed[:, 1:, :]        # (1, N_orig, D)
            D = patch_pos.shape[-1]
            N = patch_pos.shape[1]
            try:
                h0 = w0 = int(round(N ** 0.5))
                if h0 * w0 != N:
                    # non-square grid — fall back to original
                    return pos_embed
            except (ValueError, TypeError):
                return pos_embed

            try:
                patch_pos = patch_pos.reshape(1, h0, w0, D).permute(0, 3, 1, 2)
                patch_pos = F.interpolate(patch_pos, size=(h, w), mode="bicubic",
                                          align_corners=False)
                patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h * w, D)
                return torch.cat([cls_pos, patch_pos], dim=1)
            except Exception:  # noqa: BLE001
                return pos_embed

    model = _ViTBackbone()

    # Load SSL / custom checkpoint
    if pretrained in ("ssl_cxr", "checkpoint"):
        if pretrained_ckpt is None:
            warnings.warn(
                f"pretrained='{pretrained}' but pretrained_ckpt is None. "
                "Using random weights.",
                stacklevel=2,
            )
        else:
            _load_vit_weights(model.vit, pretrained_ckpt)

    return model


# ---------------------------------------------------------------------------
# Weight loading helper (DINO / SSL checkpoints)
# ---------------------------------------------------------------------------

def _load_vit_weights(vit: Any, ckpt_path: str) -> None:
    """
    Load encoder weights from a DINO or MoCo-v3 ViT checkpoint.
    Handles ``student.``, ``teacher.``, ``module.`` key prefixes.
    """
    _require(_TORCH_AVAILABLE, "torch")
    import torch  # type: ignore[import-untyped]

    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        ckpt = torch.load(str(path), map_location="cpu")
    except Exception as exc:
        raise OSError(f"Failed to load ViT checkpoint {path}: {exc}") from exc

    # DINO stores teacher/student; prefer teacher (more stable)
    for key in ("teacher", "student", "state_dict", "model"):
        if key in ckpt:
            state = ckpt[key]
            break
    else:
        state = ckpt

    cleaned: dict[str, Any] = {}
    for k, v in state.items():
        for prefix in ("module.", "backbone.", "encoder.", "vit."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        cleaned[k] = v

    missing, unexpected = vit.load_state_dict(cleaned, strict=False)
    if missing:
        warnings.warn(f"ViT: {len(missing)} missing keys: {missing[:3]}…", stacklevel=3)
    if unexpected:
        warnings.warn(f"ViT: {len(unexpected)} unexpected keys.", stacklevel=3)


# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers
# ---------------------------------------------------------------------------

def freeze_vit(model: Any, freeze_blocks: int | None = None) -> None:
    """
    Freeze ViT weights.  If ``freeze_blocks`` is set, only freeze the
    first N transformer blocks (partial freeze for gradual fine-tuning).
    """
    _require(_TORCH_AVAILABLE, "torch")
    if freeze_blocks is None:
        for p in model.vit.parameters():
            p.requires_grad = False
    else:
        # Freeze patch embed, pos embed, cls token
        for name, p in model.vit.named_parameters():
            if any(name.startswith(k) for k in ("patch_embed", "pos_embed", "cls_token")):
                p.requires_grad = False
        # Freeze first N blocks
        if hasattr(model.vit, "blocks"):
            for i, blk in enumerate(model.vit.blocks):
                if i < freeze_blocks:
                    for p in blk.parameters():
                        p.requires_grad = False


def unfreeze_vit(model: Any) -> None:
    """Unfreeze all ViT parameters."""
    _require(_TORCH_AVAILABLE, "torch")
    for p in model.vit.parameters():
        p.requires_grad = True
    if hasattr(model, "stem"):
        for p in model.stem.parameters():
            p.requires_grad = True
