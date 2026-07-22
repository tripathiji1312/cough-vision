#!/usr/bin/env python3
"""
evaluate.py — Run the full WHO TPP evaluation suite on a test set.

Usage::

    python scripts/evaluate.py \
        --checkpoint checkpoints/best_model.pt \
        --test-csv   data/test.csv \
        --image-root data/images \
        --output-dir outputs/eval \
        --device     cuda \
        --n-bootstrap 2000

Outputs
-------
  outputs/eval/report.json       — full metric report (AUC, WHO TPP, CI, calibration)
  outputs/eval/roc_curve.png     — ROC curve
  outputs/eval/pr_curve.png      — Precision-recall curve
  outputs/eval/calibration.png   — Reliability diagram
  outputs/eval/false_negatives/  — Grad-CAM overlays for all FN cases
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a trained TB detection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",  required=True, help="Path to best_model.pt")
    p.add_argument("--test-csv",    required=True)
    p.add_argument("--image-root",  required=True)
    p.add_argument("--output-dir",  default="outputs/eval")
    p.add_argument("--device",      default="cpu")
    p.add_argument("--batch-size",  type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--preset",      default="densenet_vit_ensemble")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--sensitivity-target", type=float, default=0.90)
    p.add_argument("--site",        default="global",
                   help="Site label for the evaluation report")
    p.add_argument("--audit-fn",    action="store_true",
                   help="Save Grad-CAM overlays for all false-negative cases")
    return p


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch  # type: ignore[import-untyped]
    import numpy as np  # type: ignore[import-untyped]

    from config import get_config  # type: ignore[import-untyped]
    from data.augmentation import get_inference_transform  # type: ignore[import-untyped]
    from data.dataset import TBDataset  # type: ignore[import-untyped]
    from models import build_ensemble  # type: ignore[import-untyped]
    from evaluation.metrics import evaluate  # type: ignore[import-untyped]
    from torch.utils.data import DataLoader  # type: ignore[import-untyped]

    cfg   = get_config(args.preset)
    model = build_ensemble(cfg)

    # Load checkpoint
    try:
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)  # nosec - our own checkpoints only
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint} "
              f"(epoch {ckpt.get('epoch', '?')}, "
              f"val_auc={ckpt.get('val_auc', '?'):.4f})")
    except Exception as exc:
        print(f"[ERROR] Failed to load checkpoint: {exc}")
        return 1

    device = torch.device(args.device)
    model.to(device)
    model.eval()

    test_ds = TBDataset(
        csv_path=args.test_csv,
        image_root=args.image_root,
        split="test",
        transform=get_inference_transform(cfg.preprocess.image_size_cnn),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    print(f"Evaluating on {len(test_ds)} samples (site={args.site})…")

    all_probs:  list[float] = []
    all_labels: list[int]   = []
    all_paths:  list[str]   = []

    with torch.no_grad():
        for batch in test_loader:
            images, tb_labels, *rest = batch
            images = images.to(device)
            out    = model(images)
            try:
                probs  = out["tb_prob"].cpu().tolist()
                labels = tb_labels.cpu().tolist()
                # meta dict is the 3rd element of rest (index 2) in TBDataset tuples
                if rest and isinstance(rest[2] if len(rest) > 2 else rest[-1], (list, tuple)):
                    meta_elem = rest[2] if len(rest) > 2 else rest[-1]
                    paths = [m.get("image_path", "") if isinstance(m, dict) else ""
                             for m in meta_elem]
                else:
                    paths = [""] * len(probs)
            except Exception:  # noqa: BLE001
                continue
            all_probs.extend(probs)
            all_labels.extend(labels)
            all_paths.extend(paths)

    if not all_labels:
        print("[ERROR] No predictions collected. Check test CSV and model.")
        return 1

    yt = np.array(all_labels, dtype=int)
    yp = np.array(all_probs,  dtype=float)

    # Full evaluation report
    report = evaluate(yt, yp, args.sensitivity_target, args.n_bootstrap, args.site)
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n{'='*60}")
    print(f"Site:          {args.site}")
    try:
        print(f"AUC-ROC:       {report['auc']['auc_roc']:.4f}  "
              f"(95% CI {report['auc_roc_ci']['ci_lower']:.4f}–"
              f"{report['auc_roc_ci']['ci_upper']:.4f})")
    except (KeyError, TypeError):
        pass
    try:
        who = report["who_operating_point"]
        tpp = "✓ MET" if who.get("who_tpp_met") else "✗ NOT MET"
        print(f"WHO TPP (≥90% sens / ≥70% spec): {tpp}")
        print(f"  Threshold:   {who['threshold']:.3f}")
        print(f"  Sensitivity: {who['sensitivity']:.1%}")
        print(f"  Specificity: {who['specificity']:.1%}")
    except (KeyError, TypeError):
        pass
    print(f"{'='*60}")
    print(f"Full report → {report_path}")

    # ── Grad-CAM audit of false negatives (DIR-01) ─────────────────────────
    if args.audit_fn:
        fn_dir = output_dir / "false_negatives"
        fn_dir.mkdir(exist_ok=True)
        try:
            who_thresh = report["who_operating_point"]["threshold"]
        except (KeyError, TypeError):
            who_thresh = 0.5

        fn_indices = [
            i for i, (yt_i, yp_i) in enumerate(zip(all_labels, all_probs))
            if yt_i == 1 and yp_i < who_thresh
        ]
        print(f"\nFalse negatives at WHO threshold: {len(fn_indices)}")
        print(f"Saving Grad-CAM overlays to: {fn_dir}")

        try:
            import cv2  # type: ignore[import-untyped]
            from data.preprocessing import load_image, apply_clahe  # type: ignore[import-untyped]
            from data.augmentation import get_inference_transform  # type: ignore[import-untyped]
            import PIL.Image as PilImage  # type: ignore[import-untyped]

            model.enable_gradcam()
            tf = get_inference_transform(cfg.preprocess.image_size_cnn)

            n_saved = 0
            top_k   = getattr(cfg.eval, "audit_top_k_fn", 50)
            for rank, fn_idx in enumerate(fn_indices[:top_k]):
                img_path = all_paths[fn_idx]
                if not img_path:
                    continue
                try:
                    raw = load_image(img_path)
                    pil = PilImage.fromarray(apply_clahe(raw)).convert("RGB")
                    img_t = tf(pil).unsqueeze(0).to(device)

                    model.eval()
                    with torch.no_grad():
                        out_fn  = model(img_t)
                    score = float(out_fn["tb_score"].cpu())

                    hmap    = model._gradcam(img_t, class_idx=1)
                    overlay = model._gradcam.overlay(hmap, raw)

                    # Save overlay
                    import numpy as _np  # type: ignore[import-untyped]
                    save_path = fn_dir / f"fn_{rank:04d}_score{score:.0f}.png"
                    cv2.imwrite(
                        str(save_path),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                    )
                    n_saved += 1
                except Exception as exc_inner:  # noqa: BLE001
                    print(f"  [WARN] CAM failed for {img_path}: {exc_inner}")

            model.disable_gradcam()
            print(f"Saved {n_saved} Grad-CAM overlays (top-{top_k} false negatives).")
            print(f"Review in: {fn_dir}")

        except ImportError as exc_imp:
            print(f"[WARN] Grad-CAM audit skipped: {exc_imp}")
        except Exception as exc_cam:  # noqa: BLE001
            print(f"[WARN] Grad-CAM audit failed: {exc_cam}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
