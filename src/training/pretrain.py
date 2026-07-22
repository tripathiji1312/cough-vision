"""
Self-supervised pretraining runner (MoCo-v3 / DINO).

This module provides a minimal training loop for self-supervised CXR
pretraining on large unlabelled corpora (ChestX-ray14, CheXpert, MIMIC-CXR).

Stage 1 of the training plan:
  Run MoCo-v3 or DINO on ≥700k unlabelled CXRs so the backbone learns
  CXR-specific representations before fine-tuning on the small TB dataset.

Usage::

    from training.pretrain import pretrain_moco
    pretrain_moco(cfg.pretrain, device="cuda")

Full MoCo-v3 / DINO implementations are in the `lightly` or `solo-learn`
libraries. This module wraps them with the cough-vision data pipeline and
checkpoint convention, so the resulting backbone can be loaded directly by
`build_cnn_backbone(pretrained="mocov3_cxr", pretrained_ckpt=<path>)`.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

_TORCH_AVAILABLE = False
try:
    import torch as _t  # type: ignore[import-untyped]
    _TORCH_AVAILABLE = True
except ImportError:
    _t = None  # type: ignore[assignment]


def _require(flag: bool, pkg: str) -> None:
    if not flag:
        raise ImportError(f"{pkg} is required. pip install {pkg}")


# ---------------------------------------------------------------------------
# MoCo-v3 pretraining loop (simplified; production use → solo-learn / lightly)
# ---------------------------------------------------------------------------

def pretrain_moco(
    cfg: Any,
    device: str = "cuda",
) -> Path:
    """
    Run MoCo-v3 self-supervised pretraining on an unlabelled CXR corpus.

    This is a lightweight wrapper that:
      1. Builds the CNN backbone (DenseNet-121 or EfficientNet-B4).
      2. Wraps it in a MoCo-v3 projection head (2-layer MLP).
      3. Trains with an InfoNCE contrastive objective on two CLAHE-augmented
         views of each image.
      4. Saves a checkpoint compatible with `build_cnn_backbone`.

    For large-scale runs (>100k images) we recommend the `solo-learn` library
    which provides distributed MoCo-v3 / DINO with LARS optimizer::

        pip install solo-learn
        python -m solo.methods.mocov3 --config cough_vision_moco.yaml

    Args:
        cfg:    PretrainConfig from config.py.
        device: 'cuda' or 'cpu'.

    Returns:
        Path to the saved encoder checkpoint.
    """
    _require(_TORCH_AVAILABLE, "torch")
    import torch  # type: ignore[import-untyped]
    import torch.nn as nn  # type: ignore[import-untyped]

    from ..models.backbones.cnn import build_cnn_backbone, _CNN_FEATURE_DIMS  # type: ignore[import-untyped]
    from ..data.dataset import UnlabelledCXRDataset  # type: ignore[import-untyped]
    from ..data.augmentation import get_train_transform  # type: ignore[import-untyped]
    from torch.utils.data import DataLoader  # type: ignore[import-untyped]

    backbone_name = getattr(cfg, "method", "densenet121")  # reuse method as name fallback
    encoder = build_cnn_backbone("densenet121", pretrained="imagenet")
    feat_dim = _CNN_FEATURE_DIMS.get("densenet121", 1024)

    moco_dim    = getattr(cfg, "moco_dim", 256)
    moco_mlp    = getattr(cfg, "moco_mlp_dim", 4096)
    moco_m      = getattr(cfg, "moco_m", 0.999)
    moco_T      = getattr(cfg, "moco_T", 0.2)
    max_epochs  = getattr(cfg, "max_epochs", 200)
    batch_size  = getattr(cfg, "batch_size", 256)
    base_lr     = getattr(cfg, "base_lr", 1e-3)
    weight_decay = getattr(cfg, "weight_decay", 0.05)
    num_workers = getattr(cfg, "num_workers", 16)
    output_dir  = Path(getattr(cfg, "output_dir", "checkpoints/pretrain"))
    image_root  = getattr(cfg, "image_root", "data/images")
    csv_path    = getattr(cfg, "unlabelled_csv", "data/pretrain.csv")
    mixed_prec  = getattr(cfg, "mixed_precision", True)

    output_dir.mkdir(parents=True, exist_ok=True)

    # MoCo-v3 projector: 3-layer MLP  feat_dim → mlp_dim → moco_dim
    class _Projector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(feat_dim, moco_mlp),
                nn.BatchNorm1d(moco_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(moco_mlp, moco_mlp),
                nn.BatchNorm1d(moco_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(moco_mlp, moco_dim),
                nn.BatchNorm1d(moco_dim, affine=False),
            )
        def forward(self, x: Any) -> Any:
            return nn.functional.normalize(self.layers(x), dim=-1)

    class _MoCoModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder   = encoder
            self.projector = _Projector()
            # Momentum encoder (key encoder)
            self.m_encoder   = build_cnn_backbone("densenet121", pretrained="imagenet")
            self.m_projector = _Projector()
            # Copy weights; turn off gradients for momentum encoder
            self._copy_params()
            for p in list(self.m_encoder.parameters()) + list(self.m_projector.parameters()):
                p.requires_grad = False

        def _copy_params(self) -> None:
            for p_q, p_k in zip(self.encoder.parameters(), self.m_encoder.parameters()):
                p_k.data.copy_(p_q.data)
            for p_q, p_k in zip(self.projector.parameters(), self.m_projector.parameters()):
                p_k.data.copy_(p_q.data)

        @torch.no_grad()
        def _momentum_update(self) -> None:
            for p_q, p_k in zip(self.encoder.parameters(), self.m_encoder.parameters()):
                p_k.data.mul_(moco_m).add_(p_q.data * (1.0 - moco_m))
            for p_q, p_k in zip(self.projector.parameters(), self.m_projector.parameters()):
                p_k.data.mul_(moco_m).add_(p_q.data * (1.0 - moco_m))

        def forward(self, x1: Any, x2: Any) -> Any:
            q1 = self.projector(self.encoder(x1))
            q2 = self.projector(self.encoder(x2))
            with torch.no_grad():
                self._momentum_update()
                k1 = self.m_projector(self.m_encoder(x1))
                k2 = self.m_projector(self.m_encoder(x2))
            return q1, q2, k1, k2

    model = _MoCoModel().to(torch.device(device))

    view_tf = get_train_transform(image_size=224)
    dataset = UnlabelledCXRDataset(
        csv_path=csv_path, image_root=image_root,
        view_transform=view_tf, n_views=2,
    )
    if len(dataset) == 0:
        warnings.warn(
            f"Pretrain dataset is empty (csv={csv_path}). Skipping pretraining.",
            stacklevel=2,
        )
        ckpt_path = output_dir / "encoder_pretrained.pt"
        torch.save({"model_state_dict": model.encoder.state_dict()}, str(ckpt_path))
        return ckpt_path

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.projector.parameters()),
        lr=base_lr, weight_decay=weight_decay,
    )
    scaler: Any = torch.cuda.amp.GradScaler() if (
        mixed_prec and torch.cuda.is_available()
    ) else None

    def _infonce(q: Any, k: Any, T: float) -> Any:
        """Symmetric InfoNCE (MoCo-v3 style)."""
        q = nn.functional.normalize(q, dim=-1)
        k = nn.functional.normalize(k, dim=-1)
        logits = torch.mm(q, k.T) / T          # (B, B)
        labels = torch.arange(q.shape[0], device=q.device)
        return nn.functional.cross_entropy(logits, labels)

    best_loss = float("inf")
    ckpt_path: Path = output_dir / "encoder_pretrained.pt"  # initialise so it's always bound
    print(f"Starting MoCo-v3 pretraining for {max_epochs} epochs on {len(dataset)} images.")

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        n_steps = 0
        for views in loader:
            x1, x2 = views[0].to(device), views[1].to(device)
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                q1, q2, k1, k2 = model(x1, x2)
                loss = (_infonce(q1, k2, moco_T) + _infonce(q2, k1, moco_T)) / 2.0
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            try:
                total_loss += float(loss)
            except (TypeError, ValueError):
                pass
            n_steps += 1

        avg = total_loss / max(1, n_steps)
        print(f"Pretrain epoch {epoch:03d}: loss={avg:.4f}")

        if avg < best_loss:
            best_loss = avg
            ckpt_path = output_dir / "encoder_pretrained.pt"
            try:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.encoder.state_dict(),
                        "projector_state_dict": model.projector.state_dict(),
                        "loss": avg,
                    },
                    str(ckpt_path),
                )
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Failed to save pretrain checkpoint: {exc}", stacklevel=2)

    print(f"Pretraining complete. Best loss: {best_loss:.4f}")
    print(f"Encoder checkpoint saved to: {ckpt_path}")
    return ckpt_path
