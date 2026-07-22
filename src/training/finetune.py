"""
Supervised fine-tuning trainer for the TBEnsemble.

Implements the two-phase strategy from the research plan:
  Phase 2a: Freeze backbones → train heads only  (freeze_epochs epochs)
  Phase 2b: Unfreeze all    → full fine-tune with discriminative LRs

Features:
  - Mixed-precision (fp16) via torch.cuda.amp
  - Gradient accumulation for effective batch size 64
  - Early stopping on val_auc_roc (patience=10)
  - CutMix / MixUp via CutMixMixUpCollator
  - Checkpoint saving (best + last)
  - Optional W&B logging
"""

from __future__ import annotations

import time
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
# Single epoch: train
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    criterion: Any,
    scheduler: Any,
    scaler: Any,
    device: Any,
    accumulation_steps: int = 2,
    epoch: int = 0,
    log_every: int = 50,
) -> dict[str, float]:
    """
    Run one training epoch with gradient accumulation and mixed precision.

    Returns:
        Dict with keys: loss, loss_tb, loss_findings, loss_active.
    """
    _require(_TORCH_AVAILABLE, "torch")
    import torch  # type: ignore[import-untyped]

    model.train()
    totals: dict[str, float] = {
        "loss": 0.0, "loss_tb": 0.0,
        "loss_findings": 0.0, "loss_active": 0.0,
    }
    n_batches = 0
    optimizer.zero_grad()

    try:
        for step, batch in enumerate(loader):
            # ----------------------------------------------------------------
            # Unpack — supports CutMixMixUpCollator (batch[1]=dict) and
            # standard TBDataset (batch[1]=int label).
            # ----------------------------------------------------------------
            if isinstance(batch[1], dict):
                images      = batch[0].to(device)
                label_dict  = batch[1]
                tb_labels   = label_dict["labels_a"].to(device)
                lam         = float(label_dict.get("lam", 1.0))
                labels_b_raw = label_dict.get("labels_b")
                labels_b    = labels_b_raw.to(device) if labels_b_raw is not None else None
                findings    = torch.zeros(images.shape[0], 6, device=device)
                active: Any = torch.full(
                    (images.shape[0],), -1, dtype=torch.long, device=device
                )
            else:
                images, tb_labels_raw, findings_raw, active_raw, *_ = batch
                images    = images.to(device)
                tb_labels = tb_labels_raw.to(device)
                findings  = findings_raw.to(device)
                active    = active_raw.to(device)
                lam       = 1.0
                labels_b  = None

            # ----------------------------------------------------------------
            # Forward + loss (mixed precision)
            # ----------------------------------------------------------------
            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                out  = model(images)
                loss_dict = criterion(
                    out, tb_labels, findings, active,
                    lam=lam, labels_b=labels_b,
                )
                loss = loss_dict["loss"] / accumulation_steps

            if not torch.isfinite(loss_dict["loss"]):
                warnings.warn(f"NaN loss at step {step}, skipping", stacklevel=2)
                optimizer.zero_grad()
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            # Accumulate metrics
            for k in totals:
                v = loss_dict.get(k)
                if v is not None:
                    try:
                        totals[k] += float(v)
                    except (TypeError, ValueError):
                        pass
            n_batches += 1

            if log_every > 0 and (step + 1) % log_every == 0:
                avg = totals["loss"] / max(1, n_batches)
                print(f"  Epoch {epoch} step {step+1}: loss={avg:.4f}")

        # ----------------------------------------------------------------
        # Flush any remaining accumulated gradients from the final partial
        # batch (fires when len(loader) % accumulation_steps != 0).
        # ----------------------------------------------------------------
        remaining = len(loader) % accumulation_steps  # type: ignore[arg-type]
        if remaining != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

    except Exception as exc:
        warnings.warn(f"Training loop interrupted at step {n_batches}: {exc}", stacklevel=2)
        raise

    return {k: v / max(1, n_batches) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Single epoch: validate
# ---------------------------------------------------------------------------

def validate_one_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
) -> dict[str, float]:
    """
    Run one validation epoch and return loss + AUC-ROC.

    Returns:
        Dict with keys: loss, auc_roc.
    """
    _require(_TORCH_AVAILABLE, "torch")
    import torch  # type: ignore[import-untyped]

    try:
        from sklearn.metrics import roc_auc_score as _roc_auc  # type: ignore[import-untyped]
        _sklearn_ok = True
    except ImportError:
        _roc_auc = None  # type: ignore[assignment]
        _sklearn_ok = False

    model.eval()
    total_loss = 0.0
    n_batches  = 0
    all_probs: list[float] = []
    all_labels: list[int]  = []

    with torch.no_grad():
        try:
            for batch in loader:
                if isinstance(batch[1], dict):
                    images = batch[0].to(device)
                    tb_labels = batch[1]["labels_a"].to(device)
                    findings  = torch.zeros(images.shape[0], 6, device=device)
                    active    = torch.full((images.shape[0],), -1, dtype=torch.long, device=device)
                else:
                    images, tb_labels, findings, active, *_ = batch
                    images    = images.to(device)
                    tb_labels = tb_labels.to(device)
                    findings  = findings.to(device)
                    active    = active.to(device)

                out       = model(images)
                loss_dict = criterion(out, tb_labels, findings, active)
                total_loss += float(loss_dict["loss"])
                n_batches  += 1

                try:
                    probs = out["tb_prob"].cpu().tolist()
                    if tb_labels.dim() > 1:
                        labs = tb_labels.argmax(dim=-1).cpu().tolist()
                    else:
                        labs = tb_labels.cpu().tolist()
                    all_probs.extend(probs)
                    all_labels.extend(labs)
                except Exception:  # noqa: BLE001
                    pass

        except Exception as exc:
            warnings.warn(f"Validation loop error: {exc}", stacklevel=2)
            raise

    avg_loss = total_loss / max(1, n_batches)
    auc = 0.5
    if _sklearn_ok and _roc_auc is not None and len(set(all_labels)) > 1:
        try:
            import numpy as np  # type: ignore[import-untyped]
            auc = float(_roc_auc(np.array(all_labels), np.array(all_probs)))
        except Exception:  # noqa: BLE001
            auc = 0.5

    return {"loss": avg_loss, "auc_roc": auc}


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def fit(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    cfg: Any,
    device: str = "cuda",
    wandb_run: Any | None = None,
) -> dict[str, Any]:
    """
    Full two-phase training loop.

    Phase 2a (epochs 0..freeze_epochs):  backbone frozen, heads only.
    Phase 2b (epochs freeze_epochs..max): full fine-tune.

    Args:
        model:        TBEnsemble nn.Module.
        train_loader: DataLoader for training split.
        val_loader:   DataLoader for validation split.
        cfg:          TrainConfig (from config.py).
        device:       'cuda' or 'cpu'.
        wandb_run:    Optional W&B run for logging.

    Returns:
        Dict with 'best_auc', 'best_epoch', 'history'.
    """
    _require(_TORCH_AVAILABLE, "torch")
    import torch  # type: ignore[import-untyped]

    from .losses import MultiTaskLoss  # type: ignore[import-untyped]
    from .scheduler import build_optimizer, cosine_schedule_with_warmup  # type: ignore[import-untyped]

    device_obj = torch.device(device)
    model.to(device_obj)

    criterion: Any = MultiTaskLoss(
        cls_weight=getattr(cfg, "cls_weight", 1.0),
        findings_weight=getattr(cfg, "findings_weight", 0.3),
        active_weight=getattr(cfg, "active_weight", 0.2),
        focal_gamma=getattr(cfg, "focal_gamma", 2.0),
        focal_alpha=getattr(cfg, "focal_alpha", 0.75),
        label_smoothing=0.1,
    )
    criterion.to(device_obj)

    max_epochs       = getattr(cfg, "max_epochs", 60)
    freeze_epochs    = getattr(cfg, "freeze_epochs", 3)
    accumulation     = getattr(cfg, "accumulation_steps", 2)
    mixed_prec       = getattr(cfg, "mixed_precision", True)
    patience         = getattr(cfg, "early_stop_patience", 10)
    output_dir       = Path(getattr(cfg, "output_dir", "checkpoints"))
    warmup_epochs    = getattr(cfg, "warmup_epochs", 5)
    backbone_lr      = getattr(cfg, "backbone_lr", 1e-5)
    head_lr          = getattr(cfg, "head_lr", 1e-3)
    weight_decay     = getattr(cfg, "weight_decay", 1e-4)
    log_every        = getattr(cfg, "log_every_n_steps", 50)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 2a: freeze backbones
    _m = model.module if hasattr(model, "module") else model
    try:
        _m.freeze_backbones()
    except AttributeError:
        pass

    optimizer = build_optimizer(model, backbone_lr, head_lr, weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps     = max_epochs * steps_per_epoch
    warmup_steps    = warmup_epochs * steps_per_epoch

    scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler: Any = torch.amp.GradScaler("cuda") if (
        mixed_prec and torch.cuda.is_available()
    ) else None

    best_auc   = 0.0
    best_epoch = 0
    no_improve = 0
    history: list[dict] = []

    for epoch in range(max_epochs):
        t0 = time.time()

        # Phase 2b transition
        if epoch == freeze_epochs:
            _m = model.module if hasattr(model, "module") else model
            try:
                _m.unfreeze_backbones()
                # Rebuild optimizer with discriminative LRs after unfreeze
                optimizer = build_optimizer(model, backbone_lr, head_lr, weight_decay)
                scheduler = cosine_schedule_with_warmup(optimizer, 0, total_steps)
                print(f"[Epoch {epoch}] Unfreezing backbones — full fine-tune begins.")
            except AttributeError:
                pass

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler,
            scaler, device_obj, accumulation, epoch, log_every,
        )
        val_metrics = validate_one_epoch(model, val_loader, criterion, device_obj)

        elapsed = time.time() - t0
        val_auc = val_metrics["auc_roc"]

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss":   val_metrics["loss"],
            "val_auc":    val_auc,
            "elapsed_s":  elapsed,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_auc={val_auc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if wandb_run is not None:
            try:
                wandb_run.log(row)
            except Exception:  # noqa: BLE001
                pass

        # Save best checkpoint
        if val_auc > best_auc:
            best_auc   = val_auc
            best_epoch = epoch
            no_improve = 0
            try:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_auc": val_auc,
                        "config": str(cfg),
                    },
                    output_dir / "best_model.pt",
                )
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Failed to save best checkpoint: {exc}", stacklevel=2)
        else:
            no_improve += 1

        # Save latest checkpoint
        try:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict()},
                output_dir / "last_model.pt",
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Failed to save latest checkpoint: {exc}", stacklevel=2)

        # Early stopping
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    return {"best_auc": best_auc, "best_epoch": best_epoch, "history": history}
