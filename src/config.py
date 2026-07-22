"""
Config module — single source of truth for all hyperparameters and paths.

Usage::

    from config import get_config, TrainConfig
    cfg = get_config("densenet_vit_ensemble")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project-level paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"
OUTPUT_DIR = ROOT / "outputs"

for _d in (DATA_DIR, CHECKPOINT_DIR, LOG_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------
@dataclass
class PreprocessConfig:
    """CLAHE + resize + normalisation settings."""

    image_size_cnn: int = 224
    """CNN branch input resolution (DenseNet-121 / EfficientNet-B4)."""

    image_size_vit: int = 384
    """ViT branch input resolution — higher to preserve subtle texture."""

    clahe_clip_limit: float = 2.5
    clahe_tile_grid: tuple[int, int] = (8, 8)
    gaussian_blur_sigma: float = 0.5

    # ImageNet normalisation (used when loading pretrained weights)
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # View selection
    prefer_pa: bool = True
    reject_laterals: bool = True

    # Quality-control thresholds
    min_lung_area_fraction: float = 0.15
    """Reject if segmented lung mask covers <15 % of the image area."""


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
@dataclass
class AugConfig:
    """Training-time augmentation — NO horizontal flip (anatomically wrong)."""

    rotation_degrees: float = 10.0
    translate_fraction: float = 0.05
    scale_range: tuple[float, float] = (0.85, 1.15)
    brightness_jitter: float = 0.2
    contrast_jitter: float = 0.2

    # Advanced
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0
    use_mixup: bool = True
    mixup_alpha: float = 0.4

    # Explicitly disabled
    horizontal_flip: bool = False  # NEVER flip — would mirror the heart
    vertical_flip: bool = False


# ---------------------------------------------------------------------------
# Segmentation (U-Net)
# ---------------------------------------------------------------------------
@dataclass
class SegmentationConfig:
    """U-Net lung-field segmentation model settings."""

    encoder: str = "resnet34"
    encoder_weights: str = "imagenet"
    in_channels: int = 1
    classes: int = 1
    activation: str = "sigmoid"
    input_size: int = 512

    pretrained_ckpt: str | None = None
    """Path to a fine-tuned U-Net checkpoint (JSRT/MS-CXR trained)."""


# ---------------------------------------------------------------------------
# Backbone — CNN branch
# ---------------------------------------------------------------------------
@dataclass
class CNNBackboneConfig:
    """DenseNet-121 or EfficientNet-B4 backbone settings."""

    name: str = "densenet121"
    """One of: 'densenet121', 'efficientnet_b4'."""

    pretrained: str = "imagenet"
    """Pretraining source: 'imagenet', 'mocov3_cxr', 'checkpoint'.
    Set to 'mocov3_cxr' and supply pretrained_ckpt once you have a
    CXR-specific checkpoint — 'imagenet' is the safe default."""

    pretrained_ckpt: str | None = None
    """Path to a MoCo-CXR checkpoint when pretrained='checkpoint'."""

    drop_rate: float = 0.2
    feature_dim: int = 1024
    """Dimension of the final GAP feature vector (1024 for DenseNet-121)."""

    freeze_epochs: int = 2
    """Freeze backbone for this many epochs before full fine-tuning (stage 2a)."""


# ---------------------------------------------------------------------------
# Backbone — ViT branch
# ---------------------------------------------------------------------------
@dataclass
class ViTBackboneConfig:
    """Conv-stem ViT-S/16 branch settings."""

    name: str = "vit_small_patch16_384"
    """timm model name — resolved at runtime."""

    pretrained: str = "imagenet"
    """Pretraining source: 'imagenet', 'ssl_cxr', 'checkpoint'.
    Set to 'ssl_cxr' and supply pretrained_ckpt once you have a
    CXR-specific SSL checkpoint — 'imagenet' is the safe default."""

    pretrained_ckpt: str | None = None

    # Conv stem replacing patch-embed linear projection
    use_conv_stem: bool = True
    conv_stem_channels: tuple[int, ...] = (32, 64, 128)
    """Channel sizes for the 3-layer Conv2D stem before patch embed."""

    drop_rate: float = 0.1
    attn_drop_rate: float = 0.0
    feature_dim: int = 384
    """ViT-S hidden dimension."""


# ---------------------------------------------------------------------------
# Feature Fusion
# ---------------------------------------------------------------------------
@dataclass
class FusionConfig:
    """Late attention-gated fusion of CNN + ViT feature vectors."""

    method: str = "attention"
    """One of: 'concat', 'attention', 'deep_layer_agg'."""

    hidden_dim: int = 512
    """Intermediate projection dimension before classifier."""

    dropout: float = 0.3


# ---------------------------------------------------------------------------
# Classification & detection heads
# ---------------------------------------------------------------------------
@dataclass
class HeadsConfig:
    """Multi-task output heads."""

    # Head 1 — binary TB classification
    tb_classes: int = 2
    label_smoothing: float = 0.1

    # Head 2 — multi-label TB findings (auxiliary, drives Grad-CAM quality)
    findings_classes: int = 6
    """Cavitation, consolidation, pleural effusion, hilar LAD, fibrosis, nodules."""

    findings_weight: float = 0.3
    """Loss weight for findings head (β in the paper formula)."""

    # Head 3 — active vs inactive TB (optional)
    active_inactive: bool = True
    active_inactive_weight: float = 0.2
    """Loss weight (γ)."""


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """End-to-end training hyperparameters."""

    # Optimiser
    optimizer: str = "adamw"
    backbone_lr: float = 1e-5
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0

    # Scheduler
    scheduler: str = "cosine_with_warmup"
    warmup_epochs: int = 5
    max_epochs: int = 60
    freeze_epochs: int = 3
    """Epochs to train heads-only before full fine-tune (stage 2a)."""

    # Batching
    batch_size: int = 32
    accumulation_steps: int = 2
    """Effective batch size = batch_size × accumulation_steps = 64."""

    mixed_precision: bool = True
    num_workers: int = 8

    # Focal loss
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75

    # Early stopping
    early_stop_patience: int = 10
    early_stop_metric: str = "val_auc_roc"

    # Multi-task loss weights
    cls_weight: float = 1.0        # α
    findings_weight: float = 0.3   # β
    active_weight: float = 0.2     # γ

    # Logging
    log_every_n_steps: int = 50
    checkpoint_metric: str = "val_auc_roc"

    # Seeds
    seed: int = 42

    # Paths
    train_csv: str = str(DATA_DIR / "train.csv")
    val_csv: str = str(DATA_DIR / "val.csv")
    test_csv: str = str(DATA_DIR / "test.csv")
    image_root: str = str(DATA_DIR / "images")

    output_dir: str = str(CHECKPOINT_DIR)

    # wandb
    wandb_project: str = "cough-vision"
    wandb_enabled: bool = True


# ---------------------------------------------------------------------------
# Self-supervised pretraining
# ---------------------------------------------------------------------------
@dataclass
class PretrainConfig:
    """MoCo-v3 / DINO pretraining on large unlabelled CXR corpus."""

    method: str = "mocov3"
    """One of: 'mocov3', 'dino'."""

    # MoCo-v3 specific
    moco_dim: int = 256
    moco_mlp_dim: int = 4096
    moco_m: float = 0.999        # momentum for key encoder
    moco_T: float = 0.2          # temperature

    # DINO specific
    dino_out_dim: int = 65536
    dino_teacher_temp: float = 0.04
    dino_student_temp: float = 0.1
    dino_momentum: float = 0.996

    # Shared
    batch_size: int = 256
    max_epochs: int = 200
    warmup_epochs: int = 10
    base_lr: float = 1e-3
    weight_decay: float = 0.05
    optimizer: str = "adamw"
    mixed_precision: bool = True
    num_workers: int = 16

    unlabelled_csv: str = str(DATA_DIR / "pretrain.csv")
    image_root: str = str(DATA_DIR / "images")
    output_dir: str = str(CHECKPOINT_DIR / "pretrain")


# ---------------------------------------------------------------------------
# Evaluation & calibration
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    """Evaluation and per-site threshold calibration settings."""

    # WHO TPP operating point
    who_sensitivity_target: float = 0.90
    who_specificity_floor: float = 0.70

    # Bootstrap confidence intervals
    n_bootstrap: int = 2000
    ci_alpha: float = 0.05

    # Score scaling (0–100 to match CAD4TB convention)
    scale_to_100: bool = True

    # Calibration
    calibration_method: str = "platt"
    """One of: 'platt', 'isotonic', 'temperature'."""

    min_calibration_samples: int = 50
    """Minimum locally labelled CXRs needed for per-site calibration."""

    # Grad-CAM audit flags
    audit_false_negatives: bool = True
    audit_top_k_fn: int = 50


# ---------------------------------------------------------------------------
# Deployment / export
# ---------------------------------------------------------------------------
@dataclass
class DeployConfig:
    """ONNX / TFLite export and inference server settings."""

    export_format: str = "onnx"
    """One of: 'onnx', 'tflite', 'torchscript'."""

    onnx_opset: int = 17
    optimize_onnx: bool = True
    quantize: bool = False
    """INT8 quantization for edge deployment."""

    target_device: str = "cpu"
    """One of: 'cpu', 'cuda', 'tensorrt', 'tflite_arm'."""

    max_inference_ms: int = 1000
    """Alert if inference exceeds 1 s on target device."""

    model_version: str = "v1.0.0"
    """Embed in output metadata for post-market drift tracking."""


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Aggregate config passed through the whole pipeline."""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    augment: AugConfig = field(default_factory=AugConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    cnn: CNNBackboneConfig = field(default_factory=CNNBackboneConfig)
    vit: ViTBackboneConfig = field(default_factory=ViTBackboneConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)


# ---------------------------------------------------------------------------
# Named presets
# ---------------------------------------------------------------------------
_PRESETS: dict[str, dict[str, Any]] = {
    # Full ensemble — best accuracy, ~2s on CPU
    "densenet_vit_ensemble": {},

    # Edge / portable device — distilled EfficientNet-B0, <1s on Jetson Orin Nano
    "edge_efficientnet": {
        "cnn": {"name": "efficientnet_b0", "feature_dim": 320},
        "vit": {"name": "vit_tiny_patch16_224"},
        "deploy": {"target_device": "tflite_arm", "quantize": True},
        "train": {"batch_size": 16},
    },

    # EfficientNet-B4 single-branch — highest single-model AUC
    "efficientnet_b4_single": {
        "cnn": {"name": "efficientnet_b4", "feature_dim": 1792},
        "fusion": {"method": "concat"},
    },
}


def get_config(preset: str = "densenet_vit_ensemble") -> Config:
    """Return a :class:`Config` with the given preset applied on top of defaults."""
    cfg = Config()
    overrides = _PRESETS.get(preset, {})
    for section, kv in overrides.items():
        sub = getattr(cfg, section)
        for k, v in kv.items():
            setattr(sub, k, v)
    return cfg
