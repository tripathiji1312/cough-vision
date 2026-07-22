"""
U-Net lung-field segmentation model.

Architecture: encoder-decoder with skip connections.
Encoder: ResNet-34 (pretrained on ImageNet → fine-tuned on JSRT/MS-CXR masks).
Target: Dice ≈ 0.96, Pixel Accuracy 97.96% on Shenzhen/Montgomery (per research plan).

The produced binary mask is used downstream to:
  1. Zero-out extra-thoracic shortcut regions (text, jewellery, pacemakers).
  2. Constrain CutMix patches to the lung field.
  3. Constrain Grad-CAM heatmaps to anatomically valid regions.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

_TORCH_AVAILABLE = False
try:
    import torch as _torch_mod          # type: ignore[import-untyped]
    import torch.nn as _nn_mod          # type: ignore[import-untyped]
    _TORCH_AVAILABLE = True
except ImportError:
    _torch_mod = None   # type: ignore[assignment]
    _nn_mod = None      # type: ignore[assignment]

_TIMM_AVAILABLE = False
try:
    import timm as _timm_mod            # type: ignore[import-untyped]
    _TIMM_AVAILABLE = True
except ImportError:
    _timm_mod = None    # type: ignore[assignment]


def _require(flag: bool, pkg: str) -> None:
    if not flag:
        raise ImportError(f"{pkg} is required. pip install {pkg}")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBnRelu(object):
    """Convenience factory — returns nn.Sequential(Conv2d, BN, ReLU)."""

    def __new__(
        cls,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        padding: int = 1,
    ) -> Any:
        _require(_TORCH_AVAILABLE, "torch")
        import torch.nn as nn  # type: ignore[import-untyped]
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(object):
    """Convenience factory — upsampling + two ConvBnRelu blocks."""

    def __new__(cls, in_ch: int, skip_ch: int, out_ch: int) -> Any:
        _require(_TORCH_AVAILABLE, "torch")
        import torch.nn as nn  # type: ignore[import-untyped]

        class _Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                self.conv1 = nn.Sequential(
                    nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
                self.conv2 = nn.Sequential(
                    nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x: Any, skip: Any) -> Any:
                x = self.upsample(x)
                # Pad if spatial sizes differ by 1 px (odd input sizes)
                if x.shape[-2:] != skip.shape[-2:]:
                    import torch.nn.functional as F  # type: ignore[import-untyped]
                    x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
                x = self.conv1(
                    __import__("torch").cat([x, skip], dim=1)
                )
                return self.conv2(x)

        return _Block()


# ---------------------------------------------------------------------------
# Main U-Net model
# ---------------------------------------------------------------------------

class LungSegmentationUNet(object):
    """
    Factory that builds a ResNet-34 encoder U-Net for binary lung-field
    segmentation.  Returns an ``nn.Module`` instance.

    Args:
        encoder_name:     timm model name used as encoder backbone.
        pretrained:       Load ImageNet weights for the encoder.
        in_channels:      Input channels (1 = greyscale; 3 = RGB).
        decoder_channels: Output channels per decoder stage (coarse→fine).

    Returns:
        An ``nn.Module`` with a ``.forward(x)`` method that accepts
        a (B, in_channels, H, W) tensor and returns a (B, 1, H, W) sigmoid
        probability map.  Call ``(output > 0.5).float()`` for binary mask.
    """

    def __new__(
        cls,
        encoder_name: str = "resnet34",
        pretrained: bool = True,
        in_channels: int = 1,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32),
    ) -> Any:
        _require(_TORCH_AVAILABLE, "torch")
        _require(_TIMM_AVAILABLE, "timm")
        import torch.nn as nn  # type: ignore[import-untyped]
        import torch  # type: ignore[import-untyped]
        import timm  # type: ignore[import-untyped]

        class _UNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()

                # Encoder — strip classifier head; keep feature_info
                self.encoder = timm.create_model(
                    encoder_name,
                    pretrained=pretrained,
                    features_only=True,
                    in_chans=in_channels,
                )
                enc_channels: list[int] = [
                    info["num_chs"] for info in self.encoder.feature_info
                ]

                # Bridge (bottleneck)
                self.bridge = nn.Sequential(
                    nn.Conv2d(enc_channels[-1], enc_channels[-1] * 2, 3, padding=1, bias=False),
                    nn.BatchNorm2d(enc_channels[-1] * 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(enc_channels[-1] * 2, enc_channels[-1], 3, padding=1, bias=False),
                    nn.BatchNorm2d(enc_channels[-1]),
                    nn.ReLU(inplace=True),
                )

                # Decoder blocks (bottom-up)
                self.decoders = nn.ModuleList()
                in_ch = enc_channels[-1]
                skip_chs = list(reversed(enc_channels[:-1]))
                for i, d_ch in enumerate(decoder_channels):
                    skip = skip_chs[i] if i < len(skip_chs) else 0
                    self.decoders.append(DecoderBlock(in_ch, skip, d_ch))
                    in_ch = d_ch

                # Final 1×1 segmentation head
                self.head = nn.Sequential(
                    nn.Conv2d(in_ch, 16, 3, padding=1, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 1, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x: Any) -> Any:
                # Encoder: returns list of feature maps [stem, s1, s2, s3, s4]
                features = self.encoder(x)
                bottleneck = self.bridge(features[-1])

                # Decoder: skip connections from encoder (high-res → low-res)
                skip_features = list(reversed(features[:-1]))
                d = bottleneck
                for i, dec in enumerate(self.decoders):
                    skip = skip_features[i] if i < len(skip_features) else torch.zeros_like(d)
                    d = dec(d, skip)

                # Upsample to input resolution if needed
                import torch.nn.functional as F  # type: ignore[import-untyped]
                if d.shape[-2:] != x.shape[-2:]:
                    d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=True)

                return self.head(d)  # (B, 1, H, W)

        return _UNet()


# ---------------------------------------------------------------------------
# Dice + BCE combined loss for segmentation training
# ---------------------------------------------------------------------------

class SegmentationLoss(object):
    """
    Factory for DiceBCELoss — standard choice for medical image segmentation.
    Returns an ``nn.Module``.
    """

    def __new__(cls, bce_weight: float = 0.5, smooth: float = 1.0) -> Any:
        _require(_TORCH_AVAILABLE, "torch")
        import torch.nn as nn  # type: ignore[import-untyped]

        class _Loss(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bce = nn.BCELoss()
                self._bce_w = bce_weight
                self._smooth = smooth

            def forward(self, pred: Any, target: Any) -> Any:
                import torch  # type: ignore[import-untyped]
                # Dice
                p_flat = pred.view(-1)
                t_flat = target.view(-1)
                intersection = (p_flat * t_flat).sum()
                dice = 1.0 - (2.0 * intersection + self._smooth) / (
                    p_flat.sum() + t_flat.sum() + self._smooth
                )
                bce = self.bce(pred, target)
                return self._bce_w * bce + (1.0 - self._bce_w) * dice

        return _Loss()


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def predict_mask(
    model: Any,
    image_tensor: Any,
    threshold: float = 0.5,
    device: str = "cpu",
) -> Any:
    """
    Run the segmentation model on a single pre-processed image tensor and
    return a binary (H, W) uint8 numpy mask.

    Args:
        model:        Trained U-Net (nn.Module).
        image_tensor: Float tensor (1, C, H, W) or (C, H, W) — normalised.
        threshold:    Binarisation threshold (default 0.5).
        device:       'cpu' or 'cuda'.

    Returns:
        uint8 numpy array (H, W) with values 0/1.
    """
    _require(_TORCH_AVAILABLE, "torch")
    import torch  # type: ignore[import-untyped]
    import numpy as np  # type: ignore[import-untyped]

    model.eval()
    try:
        with torch.no_grad():
            x = image_tensor.to(device)
            if x.dim() == 3:
                x = x.unsqueeze(0)
            prob = model(x)           # (1, 1, H, W)
            mask = (prob[0, 0] > threshold).cpu().numpy().astype(np.uint8)
        return mask
    except Exception as exc:
        raise RuntimeError(f"Segmentation inference failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_segmentation_checkpoint(
    model: Any,
    optimizer: Any,
    epoch: int,
    val_dice: float,
    path: str | Path,
) -> None:
    """Save a U-Net checkpoint dict."""
    _require(_TORCH_AVAILABLE, "torch")
    import torch  # type: ignore[import-untyped]
    try:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
            },
            str(path),
        )
    except Exception as exc:
        raise OSError(f"Failed to save checkpoint to {path}: {exc}") from exc


def load_segmentation_checkpoint(
    model: Any,
    path: str | Path,
    device: str = "cpu",
) -> dict:
    """Load a U-Net checkpoint; returns metadata dict."""
    _require(_TORCH_AVAILABLE, "torch")
    import torch  # type: ignore[import-untyped]
    try:
        ckpt = torch.load(str(path), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        return {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    except Exception as exc:
        raise OSError(f"Failed to load checkpoint from {path}: {exc}") from exc
