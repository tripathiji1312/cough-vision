# cough-vision 🫁

**A clinically deployable deep-learning system for pulmonary TB detection from chest X-rays.**

> WHO Target Product Profile target: **≥90% sensitivity at ≥70% specificity** (triage use).

---

## Architecture

```
Raw CXR (DICOM/PNG)
        ↓
QC + CLAHE + Gaussian denoise
        ↓
U-Net lung-field segmentation  →  lung mask
        ↓
Crop & mask to lung ROI
        ↓
┌─────────────────────────────────────────────────┐
│  CNN Branch                  ViT Branch          │
│  DenseNet-121 / EfficientNet-B4    Conv-stem      │
│  (MoCo-CXR pretrained)       ViT-S/16            │
│  → (B, 1024) GAP features    → (B, 384) CLS tok  │
└──────────────┬──────────────────────┬────────────┘
               └──────── Attention-gated fusion ────┘
                                  ↓
                    Head 1: TB binary classification
                    Head 2: 6-class findings (Grad-CAM)
                    Head 3: Active vs inactive TB
                                  ↓
                    Score 0–100 + Grad-CAM overlay
                                  ↓
                    Per-site threshold calibration
                                  ↓
                    Clinical report + Xpert referral
```

## Project Structure

```
cough-vision/
├── plan/
│   └── README.md              # Full research plan + architecture rationale
├── src/
│   ├── config.py              # All hyperparameters and presets
│   ├── data/
│   │   ├── preprocessing.py   # CLAHE, lung masking, DICOM decoding
│   │   ├── augmentation.py    # CutMix, MixUp, NO horizontal flip
│   │   └── dataset.py         # TBDataset, UnlabelledCXRDataset, split utils
│   ├── models/
│   │   ├── __init__.py        # TBEnsemble (build_ensemble)
│   │   ├── segmentation.py    # U-Net lung-field segmentation
│   │   ├── heads.py           # Multi-task heads + score scaling
│   │   ├── interpretability.py # Grad-CAM, PC-CAM, overlay
│   │   └── backbones/
│   │       ├── cnn.py         # DenseNet-121 / EfficientNet-B4
│   │       ├── vit.py         # Conv-stem ViT-S/16
│   │       └── fusion.py      # Attention-gated feature fusion
│   ├── training/
│   │   ├── losses.py          # FocalLoss, AsymmetricLoss, MultiTaskLoss
│   │   ├── scheduler.py       # Cosine warmup + discriminative LRs
│   │   └── finetune.py        # Two-phase training loop
│   ├── evaluation/
│   │   ├── metrics.py         # AUC, WHO TPP, bootstrap CIs, ECE
│   │   └── calibration.py     # Platt / isotonic per-site calibration
│   └── deployment/
│       ├── export.py          # ONNX export + quantisation
│       └── inference.py       # InferenceEngine, TBPrediction, drift monitor
├── scripts/
│   ├── train.py               # CLI: train the ensemble
│   ├── evaluate.py            # CLI: full WHO evaluation suite
│   └── deploy.py              # CLI: export → calibrate → infer
├── tests/
│   ├── test_preprocessing.py
│   └── test_evaluation.py
└── requirements.txt
```

## Quick Start

### 1. Install

```bash
# Installs all runtime deps (torch, timm, opencv, sklearn, onnx, etc.)
pip install -e .

# With dev tools (pytest, black, mypy, isort):
pip install -e ".[dev]"

# With W&B training logging:
pip install -e ".[dev,train]"
```

### 1b. Prepare data CSVs

Before training you need `data/train.csv`, `data/val.csv`, and `data/test.csv`.
The schema is documented in `DATASETS.md`. Use the Kaggle notebook
(`notebooks/train_kaggle.ipynb`) which auto-generates them from the raw datasets,
or build them manually following the schema in `README.md#data-requirements`.

### 2. Train

```bash
python scripts/train.py \
  --preset      densenet_vit_ensemble \
  --train-csv   data/train.csv \
  --val-csv     data/val.csv \
  --image-root  / \
  --output-dir  checkpoints \
  --device      cuda \
  --epochs      60 \
  --wandb
```

### 3. Evaluate (WHO TPP report)

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --test-csv   data/test.csv \
  --image-root data/images \
  --output-dir outputs/eval \
  --site       shenzhen \
  --audit-fn        # saves Grad-CAM overlays for all false negatives
```

### 4. Deploy

```bash
# Export to ONNX
python scripts/deploy.py export \
  --checkpoint checkpoints/best_model.pt \
  --output     deploy/model.onnx \
  --optimize --quantize

# Calibrate on 50–200 local samples
python scripts/deploy.py calibrate \
  --onnx       deploy/model.onnx \
  --cal-csv    data/site_cal.csv \
  --image-root data/images \
  --output     deploy/calibrator.json

# Single-image inference
python scripts/deploy.py infer \
  --onnx       deploy/model.onnx \
  --calibrator deploy/calibrator.json \
  --image      patient_cxr.png
```

### 5. Python API

```python
from deployment.inference import InferenceEngine
from evaluation.calibration import PlattCalibrator

cal    = PlattCalibrator.load("deploy/calibrator.json")
engine = InferenceEngine.from_onnx("deploy/model.onnx", calibrator=cal)
result = engine.predict("patient_cxr.dicom")

print(f"TB Score:    {result.tb_score:.1f}/100")
print(f"TB Positive: {result.tb_positive}")
print(f"Latency:     {result.inference_ms:.1f} ms")
# → "⚠ Confirmatory Xpert MTB/RIF testing is mandatory before treatment."
```

---

## Data Requirements

| Dataset | Source | Size | Role |
| --- | --- | --- | --- |
| **Shenzhen** | NIH | 662 | Benchmark |
| **Montgomery** | NIH | 138 | External validation |
| **NIH ChestX-ray14** | NIH | ~112k | Negative mining / pretraining |
| **CheXpert / MIMIC-CXR** | Stanford / MIT | ~600k | Self-supervised pretraining |
| **India TB cohort** | India | 155+ | High-burden domain coverage |

CSV format (`data/train.csv`):

```
image_path,tb_label,findings_label,active_inactive_label,site,view_position,split,mask_path
images/SZ_001.png,1,"1,0,0,0,0,0",1,shenzhen,PA,train,masks/SZ_001.png
images/MC_001.png,0,"0,0,0,0,0,0",-1,montgomery,PA,train,
```

---

## Key Design Decisions (from research plan)

| Decision | Reason |
| --- | --- |
| **No horizontal flip augmentation** | Places heart on wrong side; corrupts spatial priors |
| **Multi-center training mandatory** | Single-source model collapsed to 52% sensitivity in India |
| **Per-site threshold calibration** | CAD4TB requires different thresholds per country |
| **Lung-field masking** | Suppresses text/scanner shortcuts that fool the model |
| **MoCo-CXR pretraining** | Beats ImageNet on CXR; improves cross-dataset transfer |
| **Multi-task findings head** | Forces backbone to learn pathology, not shortcuts |
| **Grad-CAM on every positive** | Mandatory for clinical trust (WHO-evaluated CAD standard) |
| **WHO TPP as primary metric** | ≥90% sens / ≥70% spec — not just AUC |

---

## Regulatory Notes

This system is intended as a **Software as a Medical Device (SaMD)** triage tool.

- Results **must** be confirmed with Xpert MTB/RIF before treatment.
- WHO (June 2025) requires independent FIND-validated evaluation against TPP.
- Supported regulatory pathways: FDA 510(k) · EU MDR IIa/IIb · CDSCO · WHO FIND.

---

## Tests

```bash
pytest tests/ -v
```

---

## License

For research and clinical validation use only. See `LICENSE` for details.
