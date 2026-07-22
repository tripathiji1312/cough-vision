#!/usr/bin/env python3
"""
train.py — CLI entry-point for TBEnsemble training.

Usage::

    python scripts/train.py \
        --preset densenet_vit_ensemble \
        --train-csv data/train.csv \
        --val-csv   data/val.csv \
        --image-root data/images \
        --output-dir checkpoints \
        --epochs 60 \
        --device cuda

Phases
------
1. (Optional) self-supervised MoCo/DINO pretraining on unlabelled CXRs.
2. Supervised multi-task fine-tuning on labelled TB data (two-phase:
   freeze-then-unfreeze backbones).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the cough-vision TB detection ensemble.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--preset",      default="densenet_vit_ensemble",
                   choices=["densenet_vit_ensemble", "edge_efficientnet",
                             "efficientnet_b4_single"],
                   help="Model preset from config.py")
    p.add_argument("--train-csv",   required=True, help="Path to training split CSV")
    p.add_argument("--val-csv",     required=True, help="Path to validation split CSV")
    p.add_argument("--image-root",  required=True, help="Root directory for images")
    p.add_argument("--mask-root",   default=None,  help="Root directory for lung masks")
    p.add_argument("--output-dir",  default="checkpoints")
    p.add_argument("--epochs",      type=int, default=60)
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--freeze-epochs", type=int, default=3,
                   help="Epochs to train heads-only before full fine-tune")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--wandb",       action="store_true", help="Enable W&B logging")
    p.add_argument("--wandb-project", default="cough-vision")
    p.add_argument("--pretrain-ckpt", default=None,
                   help="MoCo/DINO checkpoint path for CNN/ViT backbone init")
    return p


def main() -> int:
    args = build_parser().parse_args()

    import torch  # type: ignore[import-untyped]
    torch.manual_seed(args.seed)

    from config import get_config  # type: ignore[import-untyped]
    cfg = get_config(args.preset)

    # Apply CLI overrides
    cfg.train.train_csv   = args.train_csv
    cfg.train.val_csv     = args.val_csv
    cfg.train.image_root  = args.image_root
    cfg.train.output_dir  = args.output_dir
    cfg.train.max_epochs  = args.epochs
    cfg.train.batch_size  = args.batch_size
    cfg.train.freeze_epochs = args.freeze_epochs
    cfg.train.num_workers = args.num_workers
    cfg.train.seed        = args.seed
    cfg.train.wandb_enabled = args.wandb
    cfg.train.wandb_project = args.wandb_project
    if args.pretrain_ckpt:
        cfg.cnn.pretrained_ckpt = args.pretrain_ckpt
        cfg.cnn.pretrained      = "checkpoint"

    # DataLoaders
    from data.augmentation import get_train_transform, get_inference_transform, CutMixMixUpCollator  # type: ignore[import-untyped]
    from data.dataset import TBDataset, compute_sample_weights  # type: ignore[import-untyped]
    from torch.utils.data import DataLoader, WeightedRandomSampler  # type: ignore[import-untyped]

    train_ds = TBDataset(
        csv_path=args.train_csv,
        image_root=args.image_root,
        split="train",
        transform=get_train_transform(
            image_size=cfg.preprocess.image_size_cnn,
            rotation_degrees=cfg.augment.rotation_degrees,
            translate_fraction=cfg.augment.translate_fraction,
            scale_range=cfg.augment.scale_range,
            brightness_jitter=cfg.augment.brightness_jitter,
            contrast_jitter=cfg.augment.contrast_jitter,
            use_random_erasing=True,
        ),
        mask_root=args.mask_root,
        return_mask=args.mask_root is not None,
    )

    # Weighted sampler for class imbalance (TB prevalence ~5-10%)
    try:
        tb_labels = [int(r.get("tb_label", 0)) for r in train_ds.records]
    except (ValueError, TypeError) as exc:
        print(f"[WARN] Could not parse tb_labels for sampler: {exc}. Using uniform weights.")
        tb_labels = [0] * len(train_ds.records)
    weights   = compute_sample_weights(tb_labels, pos_weight=5.0)
    sampler   = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=CutMixMixUpCollator(n_classes=2) if cfg.augment.use_cutmix else None,
    )

    val_ds = TBDataset(
        csv_path=args.val_csv,
        image_root=args.image_root,
        split="val",
        transform=get_inference_transform(image_size=cfg.preprocess.image_size_cnn),
        mask_root=args.mask_root,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    # Build model
    from models import build_ensemble  # type: ignore[import-untyped]
    model = build_ensemble(cfg)

    # W&B
    wandb_run = None
    if args.wandb:
        try:
            import wandb  # type: ignore[import-untyped]
            wandb_run = wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"{args.preset}_{args.epochs}ep",
            )
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] W&B init failed: {e}. Continuing without W&B.")

    # Train
    from training.finetune import fit  # type: ignore[import-untyped]
    results = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg.train,
        device=args.device,
        wandb_run=wandb_run,
    )

    print(f"\n✓ Training complete.")
    print(f"  Best val AUC: {results['best_auc']:.4f} at epoch {results['best_epoch']}")
    print(f"  Checkpoints saved to: {args.output_dir}")

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:  # noqa: BLE001
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
