"""
cough-vision — Pulmonary TB Detection System
=============================================
A clinically deployable deep-learning pipeline for TB triage from CXR images.

Implements the hybrid CNN+ViT ensemble described in the research plan:
  - U-Net lung-field segmentation
  - DenseNet-121 / EfficientNet-B4 CNN branch (MoCo-CXR pretrained)
  - Conv-stem ViT-S branch
  - Late feature fusion with attention gating
  - Multi-task heads: TB classification + abnormality localization + active/inactive
  - Grad-CAM interpretability layer
  - Per-site threshold calibration
  - ONNX / TFLite export

WHO TPP target: ≥90% sensitivity at ≥70% specificity.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cough-vision")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

__all__ = [
    "__version__",
]
