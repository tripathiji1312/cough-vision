# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`cough-vision` is a clinically deployable deep-learning pipeline for pulmonary TB detection from chest X-rays. WHO Target Product Profile: ≥90% sensitivity at ≥70% specificity (triage use). Intended as Software as a Medical Device (SaMD).

## Commands

```bash
# Install
pip install -e .                    # runtime only
pip install -e ".[dev,train]"       # dev tools + W&B

# Tests
pytest tests/ -v                    # full suite
pytest tests/test_models_smoke.py   # single file

# Format / lint / typecheck
black src/ tests/ scripts/
isort src/ tests/ scripts/
flake8 src/ tests/ scripts/ --max-line-length 100
mypy src/ --ignore-missing-imports

# Train (CLI)
python scripts/train.py \
  --preset densenet_vit_ensemble \
  --train-csv data/train.csv --val-csv data/val.csv \
  --image-root / --output-dir checkpoints \
  --epochs 30 --device cuda --wandb

# Evaluate
python scripts/evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --test-csv data/test.csv --image-root / \
  --output-dir outputs/eval --audit-fn

# Deploy
python scripts/deploy.py export --checkpoint checkpoints/best_model.pt --output deploy/model.onnx
python scripts/deploy.py calibrate --onnx deploy/model.onnx --cal-csv data/site_cal.csv --image-root / --output deploy/calibrator.json
python scripts/deploy.py infer --onnx deploy/model.onnx --calibrator deploy/calibrator.json --image patient.png
```

## Architecture

### Entry points
- `src/config.py` — single source of truth for all hyperparameters. Three named presets: `densenet_vit_ensemble` (33M params, default), `edge_efficientnet` (10M, INT8), `efficientnet_b4_single` (48M, highest AUC). Access via `cfg = get_config("densenet_vit_ensemble")`.
- `src/models/__init__.py` — `build_ensemble(cfg)` factory returns `TBEnsemble`. Forward returns a dict: `tb_logits`, `tb_prob`, `tb_score` (0–100), `findings_logits`, `active_logits`, `gradcam`.
- `notebooks/train_kaggle.ipynb` — Kaggle training notebook with DDP worker, W&B logging, per-epoch checkpoint artifacts. The actual training loop lives in `ddp_worker()` (Cell 10), not in `finetune.fit()`.

### Data pipeline
- `src/data/preprocessing.py` — DICOM/PNG loading, CLAHE (clip=2.5, tile 8×8), Gaussian denoise (σ=0.5), lung masking, QC gates.
- `src/data/augmentation.py` — CutMix/MixUp collator, deterministic inference transforms. **No horizontal flip anywhere** (anatomically impossible; confirmed harmful in 2026 domain-shift study).
- `src/data/dataset.py` — `TBDataset` (CSV-driven multi-task labels), `stratified_split()` by `(site, tb_label)`. Split is sample-level, not patient-level — val AUC may be inflated for datasets with multiple images per patient.

### Model
- CNN backbone (`densenet121` or `efficientnet_b*`) → 1024-dim features
- ViT-S/16-384 with Conv-stem (3-layer: 32→64→128 channels; preserves texture, replaces linear patch embed) → 384-dim features
- Attention-gated fusion (hidden=512) → `MultiTaskHead` (3 heads: TB binary + 6 findings + active/inactive)
- `freeze_backbones()` / `unfreeze_backbones()` methods used for two-phase training

### Two-phase training
Phase 2a (epochs 0–`freeze_epochs`): backbones frozen, heads only. Phase 2b: full fine-tune with discriminative LRs (backbone_lr=1e-5, head_lr=1e-3, cosine schedule, no warmup restart).

### Evaluation / deployment
- Primary metric is **WHO TPP** (≥90% sens / ≥70% spec), not just AUC. `find_who_threshold()` in `src/evaluation/metrics.py`.
- Per-site threshold calibration is mandatory before deployment — raw scores are uncalibrated.
- ONNX export at opset 17 with optional INT8 dynamic quantization.
- `InferenceEngine.from_onnx()` wraps ONNX Runtime; includes drift monitoring (rolling mean shift >5 points triggers warning).

## Code conventions

- `from __future__ import annotations` at top of every source file.
- Lazy import guards for all optional deps (torch, cv2, timm, sklearn, onnx) — pattern is `_TORCH_AVAILABLE = False; try: import torch; _TORCH_AVAILABLE = True; except ImportError: pass`.
- Use `Any` type hints for uninstalled-dep types (not `TYPE_CHECKING` blocks).
- `torch.load` must pass `weights_only=True` in `src/`. Use `weights_only=False` only in `scripts/` with a `# nosec` comment.
- Intra-package imports use relative imports (`from .preprocessing import load_image`). Scripts use `sys.path.insert(0, str(Path(__file__).parent.parent / "src"))`.
- `conftest.py` at repo root inserts `src/` on `sys.path` for pytest — no per-file path hacks in tests.
- Line length: 100. Black + isort (profile=black).

## Key design decisions — do not change without discussion

| Decision | Reason |
|---|---|
| No horizontal flip | Anatomically impossible; confirmed harmful in domain-shift study (sensitivity collapsed to 52%) |
| Per-site threshold calibration | Skipping is a safety hazard for SaMD |
| Lung-field masking | Suppresses scanner shortcuts (text, jewellery, pacemakers) |
| Multi-task findings head | Forces backbone to learn pathology not shortcuts |
| `scale_to_cad4tb` is LINEAR | NOT the real CAD4TB non-linear mapping; see heads.py docstring |

## Open items

- DIR-01: Grad-CAM wiring into `InferenceEngine` (deployment/inference.py)
- DIR-02: MoCo pretraining path commitment (pretrain.py exists but not integrated into default workflow)
- DIR-03: Dataset-prep script (currently done in notebook Cell 4–5)
