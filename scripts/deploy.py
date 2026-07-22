#!/usr/bin/env python3
"""
deploy.py — Export a trained TBEnsemble to ONNX and run per-site calibration.

Usage::

    # Export to ONNX and optionally quantise
    python scripts/deploy.py export \
        --checkpoint checkpoints/best_model.pt \
        --output     deploy/model.onnx \
        --quantize

    # Calibrate on local site data
    python scripts/deploy.py calibrate \
        --onnx       deploy/model.onnx \
        --cal-csv    data/site_cal.csv \
        --image-root data/images \
        --output     deploy/calibrator.json \
        --method     platt

    # Single-image inference (smoke-test)
    python scripts/deploy.py infer \
        --onnx         deploy/model.onnx \
        --calibrator   deploy/calibrator.json \
        --image        sample.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def cmd_export(args: argparse.Namespace) -> int:
    import torch  # type: ignore[import-untyped]
    from config import get_config  # type: ignore[import-untyped]
    from models import build_ensemble  # type: ignore[import-untyped]
    from deployment.export import export_onnx, optimise_onnx, quantise_onnx_dynamic  # type: ignore[import-untyped]

    cfg   = get_config(args.preset)
    model = build_ensemble(cfg)

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)  # nosec - our own checkpoints only
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint (epoch={ckpt.get('epoch','?')})")
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    onnx_path = export_onnx(model, args.output, model_version=args.model_version)

    if args.optimize:
        onnx_path = optimise_onnx(onnx_path)

    if args.quantize:
        quantise_onnx_dynamic(onnx_path,
                               onnx_path.with_name(onnx_path.stem + "_int8.onnx"))

    print(f"✓ Deployment artefacts ready in: {Path(args.output).parent}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    import numpy as np  # type: ignore[import-untyped]
    from data.augmentation import get_inference_transform  # type: ignore[import-untyped]
    from data.dataset import TBDataset  # type: ignore[import-untyped]
    from deployment.inference import InferenceEngine  # type: ignore[import-untyped]
    from evaluation.calibration import build_calibrator  # type: ignore[import-untyped]
    from evaluation.metrics import find_who_threshold  # type: ignore[import-untyped]
    from torch.utils.data import DataLoader  # type: ignore[import-untyped]

    engine = InferenceEngine.from_onnx(args.onnx, device="cpu")

    cal_ds = TBDataset(
        csv_path=args.cal_csv,
        image_root=args.image_root,
        split="train",          # calibration data re-uses "train" split label
        transform=get_inference_transform(224),
    )
    loader = DataLoader(cal_ds, batch_size=1, shuffle=False, num_workers=2)

    all_scores: list[float] = []
    all_labels: list[int]   = []

    for batch in loader:
        images, tb_labels, *_ = batch
        try:
            img = images[0].numpy().transpose(1, 2, 0)  # CHW → HWC
            result = engine.predict(img)
            all_scores.append(result.tb_prob)
            all_labels.append(int(tb_labels[0]))
        except Exception:  # noqa: BLE001
            continue

    if len(set(all_labels)) < 2:
        print("[ERROR] Calibration data has only one class. Aborting.")
        return 1

    cal = build_calibrator(args.method)
    cal.fit(
        np.array(all_scores),
        np.array(all_labels),
        sensitivity_target=args.sensitivity_target,
    )

    cal.save(args.output)
    print(f"✓ Calibrator saved → {args.output}")
    print(f"  Site threshold: {cal.threshold_:.4f}")

    who = find_who_threshold(
        np.array(all_labels),
        cal.predict_proba(np.array(all_scores)),
        sensitivity_target=args.sensitivity_target,
    )
    tpp = "✓ MET" if who["who_tpp_met"] else "✗ NOT MET"
    print(f"  WHO TPP on cal set: {tpp}  "
          f"(sens={who['sensitivity']:.1%}, spec={who['specificity']:.1%})")
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    from deployment.inference import InferenceEngine  # type: ignore[import-untyped]
    from evaluation.calibration import PlattCalibrator  # type: ignore[import-untyped]

    cal = None
    if args.calibrator:
        try:
            cal = PlattCalibrator.load(args.calibrator)
            print(f"Calibrator loaded (threshold={cal.threshold_:.4f})")
        except Exception as exc:
            print(f"[WARN] Could not load calibrator: {exc}. Using raw scores.")

    engine = InferenceEngine.from_onnx(args.onnx, calibrator=cal, device="cpu")

    result = engine.predict(args.image)
    print(f"\n{'='*50}")
    print(f"Image: {args.image}")
    print(f"TB Score (0-100): {result.tb_score:.1f}")
    print(f"TB Positive:      {result.tb_positive}")
    print(f"Threshold used:   {result.site_threshold:.4f}")
    print(f"Inference time:   {result.inference_ms:.1f} ms")
    if result.findings:
        print("Findings:")
        for k, v in result.findings.items():
            print(f"  {k:<20} {v:.3f}")
    if result.tb_positive:
        print(f"\n⚠  {result.referral_note}")
    print(f"{'='*50}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="deploy.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- export ---
    pe = sub.add_parser("export")
    pe.add_argument("--checkpoint",    required=True)
    pe.add_argument("--output",        default="deploy/model.onnx")
    pe.add_argument("--preset",        default="densenet_vit_ensemble")
    pe.add_argument("--optimize",      action="store_true")
    pe.add_argument("--quantize",      action="store_true")
    pe.add_argument("--model-version", default="v1.0.0")

    # --- calibrate ---
    pc = sub.add_parser("calibrate")
    pc.add_argument("--onnx",       required=True)
    pc.add_argument("--cal-csv",    required=True)
    pc.add_argument("--image-root", required=True)
    pc.add_argument("--output",     default="deploy/calibrator.json")
    pc.add_argument("--method",     default="platt", choices=["platt", "isotonic"])
    pc.add_argument("--sensitivity-target", type=float, default=0.90)

    # --- infer ---
    pi = sub.add_parser("infer")
    pi.add_argument("--onnx",       required=True)
    pi.add_argument("--image",      required=True)
    pi.add_argument("--calibrator", default=None)

    args = p.parse_args()

    if args.cmd == "export":
        return cmd_export(args)
    if args.cmd == "calibrate":
        return cmd_calibrate(args)
    if args.cmd == "infer":
        return cmd_infer(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
