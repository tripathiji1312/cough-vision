# cough-vision — Dataset & Training Guide

> Complete reference for datasets, preprocessing decisions, CSV schema, training
> configuration, and evaluation benchmarks. Read this before starting any training run.

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Downloading Datasets](#2-downloading-datasets)
3. [CSV Schema](#3-csv-schema)
4. [Preprocessing Decisions](#4-preprocessing-decisions)
5. [Augmentation Policy](#5-augmentation-policy)
6. [Training Configuration](#6-training-configuration)
7. [Evaluation Benchmarks](#7-evaluation-benchmarks)
8. [Known Issues & Pitfalls](#8-known-issues--pitfalls)
9. [Checklist Before Training](#9-checklist-before-training)

---

## 1. Dataset Overview

### Primary Training Datasets

| Dataset | Geography | Size | TB+ | TB− | Scanner | View | License | Kaggle Slug |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Shenzhen Hospital CXR** | China | 662 | 336 | 326 | Mixed | PA + AP | Public (NIH) | `kmader/pulmonary-chest-xray-abnormalities` |
| **Montgomery County CXR** | USA | 138 | 58 | 80 | Philips | PA | Public (NIH) | `kmader/pulmonary-chest-xray-abnormalities` |
| **TBX11K** | Multi-national | 11,200 | 1,228 | 9,972 | Various | PA | CC-BY-4.0 | `usefulsensors/tbx11k` |

### Supplementary / Negative Mining

| Dataset | Geography | Size | Role | Kaggle Slug |
| --- | --- | --- | --- | --- |
| **NIH ChestX-ray14** | USA | 112,120 | Negative mining + SSL pretraining | `nih-chest-xrays/data` |
| **CheXpert** | USA | 224,316 | SSL pretraining | (Stanford — request access) |
| **MIMIC-CXR-JPG** | USA | 377,110 | SSL pretraining | (PhysioNet — credentialled) |
| **VinBigData Chest X-ray** | Vietnam | 18,000 | SE-Asia domain coverage | `awsaf49/vinbigdata-chest-xray-resized-png` |
| **RSNA Pneumonia** | USA | 30,000 | Extra negatives | `rsna-pneumonia-detection-challenge` |

### Why multi-center is non-negotiable

A 2026 multi-national external validation study showed that a DenseNet-121 trained **only** on Shenzhen:

- AUC 0.889 internally → **AUC 0.762** on Montgomery (US low-burden)
- Sensitivity **collapsed to 52.3%** on an Indian high-burden cohort
- Specificity **collapsed to 43.7%** on a US general-population cohort

The root cause: the model learned scanner-specific haze patterns, not TB pathology. **Multi-center training + lung masking + per-site calibration are the three mandatory fixes.**

---

## 2. Downloading Datasets

### Kaggle CLI (recommended)

```bash
# Install Kaggle CLI
pip install kaggle

# Set up credentials (~/.kaggle/kaggle.json)
# Get your API token from: https://www.kaggle.com/settings

# Shenzhen + Montgomery (primary TB benchmark)
kaggle datasets download -d kmader/pulmonary-chest-xray-abnormalities -p data/raw/

# TBX11K (largest weakly-labelled TB dataset)
kaggle datasets download -d usefulsensors/tbx11k -p data/raw/

# NIH ChestX-ray14 (negative mining + pretraining)
kaggle datasets download -d nih-chest-xrays/data -p data/raw/

# VinBigData (SE-Asia coverage)
kaggle datasets download -d awsaf49/vinbigdata-chest-xray-resized-png -p data/raw/

# Unzip all
for f in data/raw/*.zip; do unzip -q "$f" -d data/raw/; done
```

### NIH Direct Download

```bash
# Shenzhen + Montgomery directly from NIH
wget -r -np -nH --cut-dirs=6 -A "*.png,*.csv" \
  https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip
wget -r -np -nH --cut-dirs=6 -A "*.png,*.csv" \
  https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip
```

### CheXpert / MIMIC-CXR (credentialled)

```bash
# CheXpert — sign the DUA at https://stanfordmlgroup.github.io/competitions/chexpert/
# MIMIC-CXR — complete CITI training + sign DUA at https://physionet.org/content/mimic-cxr-jpg/

# After access granted:
wget -r -N -c -np --user YOUR_PHYSIONET_USER --ask-password \
  https://physionet.org/files/mimic-cxr-jpg/2.0.0/ -P data/raw/mimic/
```

---

## 3. CSV Schema

All datasets must be converted to a unified CSV before training. The `data/dataset.py` module reads this format exclusively.

### Required columns

```
image_path, tb_label, findings_label, active_inactive_label,
site, view_position, split
```

| Column | Type | Values | Notes |
| --- | --- | --- | --- |
| `image_path` | `str` | Absolute or relative path | Relative to `--image-root` in CLI |
| `tb_label` | `int` | `0` = Normal, `1` = TB | Required |
| `findings_label` | `str` | `"1,0,0,0,0,0"` (6 flags) | Order: cavitation, consolidation, pleural effusion, hilar LAD, fibrosis, nodule. Use `"0,0,0,0,0,0"` when unknown |
| `active_inactive_label` | `int` | `1` = Active, `0` = Inactive/Normal, `-1` = Unknown | `-1` rows are masked out of the active/inactive CE loss |
| `site` | `str` | e.g. `shenzhen`, `montgomery`, `india_aiims` | Used in stratified split and per-site calibration |
| `view_position` | `str` | `PA`, `AP`, `LL`, `RL` | AP views shift score calibration; laterals should be excluded |
| `split` | `str` | `train`, `val`, `test`, `pretrain` | Assigned by `stratified_split()` or manually |

### Optional columns

| Column | Type | Notes |
| --- | --- | --- |
| `mask_path` | `str` | Path to pre-computed lung mask (binary PNG). If absent, segmentation runs on-the-fly |
| `patient_id` | `str` | Used to prevent data leakage — ensure no patient appears in both train and test |
| `xpert_result` | `str` | `positive`, `negative`, `not_done` — microbiological reference standard |
| `hiv_status` | `str` | `positive`, `negative`, `unknown` — required for subgroup analysis |
| `age_band` | `str` | `0-14`, `15-29`, `30-44`, `45-59`, `60+` |

### Example rows

```csv
image_path,tb_label,findings_label,active_inactive_label,site,view_position,split
/data/shenzhen/CHNCXR_0001_1.png,1,"1,0,1,0,0,0",1,shenzhen,PA,train
/data/montgomery/MCUCXR_0001_0.png,0,"0,0,0,0,0,0",-1,montgomery,PA,val
/data/tbx11k/imgs/train/00001.png,1,"0,0,0,0,0,1",-1,tbx11k,PA,train
```

### Dataset-specific label extraction

#### Shenzhen

Filename encodes label: `CHNCXR_XXXX_0.png` = Normal, `CHNCXR_XXXX_1.png` = TB.

```python
tb_label = int(fname.split('_')[-1].replace('.png', ''))
```

#### Montgomery

Same convention: `MCUCXR_XXXX_0.png` = Normal, `MCUCXR_XXXX_1.png` = TB.

#### TBX11K

`labels.csv` column `label` contains: `tb` / `sick_but_non-tb` / `healthy`.

```python
tb_label = 1 if row['label'] == 'tb' else 0
```

#### NIH ChestX-ray14

No TB class — use as **negative** samples only. Filter rows where no finding is `Infiltration`, `Consolidation`, or `Effusion` to avoid confounders.

```python
tb_label = 0  # always Normal for NIH CXR14
```

---

## 4. Preprocessing Decisions

Every decision here was validated against published TB CXR literature. Do not change without re-benchmarking.

### CLAHE (Contrast-Limited Adaptive Histogram Equalisation)

```python
clahe_clip_limit = 2.5     # range 2.0–3.0; higher = more local contrast
clahe_tile_grid  = (8, 8)  # 8×8 tiles over 224px → ~28px per tile
```

**Why**: CLAHE is the single most-cited preprocessing step in TB CXR literature. It substantially increases the visibility of cavitations, consolidations, and pleural effusions by independently equalising local contrast tiles. Clip limit 2.5 prevents noise amplification while maximising pathology contrast.

### Gaussian denoising

```python
gaussian_sigma = 0.5  # kernel size = 4 px; σ ≤ 1.0 preserves texture
```

**Why**: Film-digitised or low-dose DICOM CXRs carry significant digitisation noise. Sigma ≤ 1.0 removes the noise floor without blurring cavitation walls (which are ~2–5 px wide at 224×224).

### Lung-field segmentation and masking

U-Net with ResNet-34 encoder, trained on JSRT + MS-CXR masks. Target Dice ≈ 0.96.

**Why masking is mandatory**: Without it, models learn scanner-specific halos, text overlays, ECG leads, jewellery, and pacemakers as TB predictors. The Montgomery–Shenzhen domain shift experiment (see §1) demonstrates this explicitly — specificity collapsed on the US dataset because the Chinese scanner's background haze became a negative predictor.

```python
min_lung_area_fraction = 0.15  # QC gate: reject if mask covers <15% of image
```

### Resizing

```python
image_size_cnn = 224  # DenseNet-121 (ImageNet standard)
image_size_vit = 384  # ViT-S/16 (higher res for subtle texture preservation)
```

**Why 384 for ViT**: Cavitations and fine nodules are typically 3–30 mm on real CXRs. At a standard chest X-ray pixel spacing of ~0.143 mm/px, a 5 mm nodule is ~35 px on the full-resolution image but only ~9 px after naive resize to 224. At 384 px, it becomes ~15 px — enough for the ViT's 16-px patch window to contain full nodule context.

### Normalisation

```python
mean = (0.485, 0.456, 0.406)  # ImageNet RGB means
std  = (0.229, 0.224, 0.225)  # ImageNet RGB stds
```

Greyscale CXRs are replicated to 3 channels before normalisation. ImageNet stats are used regardless of pretrained source — they remain a stable anchor even after MoCo-CXR pretraining.

---

## 5. Augmentation Policy

### Permitted transforms

| Transform | Parameters | Rationale |
| --- | --- | --- |
| `RandomAffine` (rotation) | ±10° | Patient positioning variation |
| `RandomAffine` (translate) | ±5% | Centering variation |
| `RandomAffine` (scale) | 0.85–1.15× | Field-of-view variation |
| `ColorJitter` (brightness) | ±0.2 | Exposure variation |
| `ColorJitter` (contrast) | ±0.2 | Contrast setting variation |
| `RandomErasing` | p=0.2, 2–10% area | Simulate occluded patches |
| **CutMix** | α=1.0, p=0.5 | Within-lung-field only |
| **MixUp** | α=0.4, p=0.5 | Soft label interpolation |

### Explicitly prohibited

| Transform | Reason |
| --- | --- |
| **Horizontal flip** | Mirrors the heart to the right side — anatomically impossible. The model would learn an inverted spatial prior for apical/basal disease distribution. Confirmed harmful in the 2026 multi-national validation study. |
| **Vertical flip** | Inverts apical vs. basal distribution — TB disproportionately affects upper lobes |
| `ColorJitter` saturation/hue | CXRs are greyscale — saturation/hue changes produce artefacts |
| Aggressive elastic deformation | Distorts lung architecture — confounds apical predominance signal |

### CutMix constraint

CutMix patches are constrained to the lung-field bounding box when a mask is available, preventing anatomically nonsensical patches (e.g., a liver patch pasted into the upper lung field).

---

## 6. Training Configuration

### Model presets

| Preset | CNN | ViT | Params | Target use case |
| --- | --- | --- | --- | --- |
| `densenet_vit_ensemble` | DenseNet-121 | ViT-S/16-384 | ~33 M | Production / highest accuracy |
| `efficientnet_b4_single` | EfficientNet-B4 | ViT-S/16-384 | ~48 M | Highest single-model AUC |
| `edge_efficientnet` | EfficientNet-B0 | ViT-Ti/16-224 | ~10 M | Edge / Jetson / TFLite |

### Recommended hyperparameters

```yaml
# Phase 2a: heads-only warm-up
freeze_epochs    : 3
backbone_lr      : 1e-5    # only matters in Phase 2b
head_lr          : 1e-3

# Phase 2b: full fine-tune
max_epochs       : 60
warmup_epochs    : 5
weight_decay     : 1e-4
batch_size       : 32       # per GPU
accumulation_steps: 2       # effective = 64

# Loss
focal_gamma      : 2.0
focal_alpha      : 0.75     # down-weight easy negatives
label_smoothing  : 0.1
cls_weight       : 1.0      # α (TB classification)
findings_weight  : 0.3      # β (6-class findings)
active_weight    : 0.2      # γ (active/inactive)

# Regularisation
early_stop_patience: 10     # on val_auc_roc
mixed_precision  : true
```

### Two-phase training rationale

**Phase 2a (freeze → heads only):**  Catastrophic forgetting is a real risk when fine-tuning large pretrained backbones on tiny TB datasets (Shenzhen has only 662 images). Freezing the backbone for the first few epochs lets the classification heads stabilise before the backbone weights are disturbed. This is especially important for the ViT branch whose self-attention patterns take longer to align.

**Phase 2b (unfreeze → discriminative LRs):** The backbone uses `lr = 1e-5` (10× lower than heads). Lower-layer features from ImageNet pretraining remain largely intact; only the top layers adapt to TB pathology. This is the standard approach from ULMFiT and is validated on medical imaging tasks.

### Hardware requirements

| Config | VRAM | Time / epoch (800 samples) | Kaggle tier |
| --- | --- | --- | --- |
| `densenet_vit_ensemble` fp16 | 12 GB | ~3 min | P100 / T4 |
| `densenet_vit_ensemble` fp16 | 16 GB | ~2 min | V100 |
| `edge_efficientnet` fp16 | 6 GB | ~1.5 min | T4 |

Kaggle provides **30h/week** of GPU time (T4 ×2 or P100 ×1). A full 60-epoch run on Shenzhen+Montgomery+TBX11K (~12k samples) takes approximately **8–10h on a T4**.

### Multi-GPU (DDP) — optional

```bash
# Kaggle provides T4 ×2 for some tiers
torchrun --nproc_per_node=2 scripts/train.py \
  --preset densenet_vit_ensemble \
  --train-csv data/train.csv \
  --val-csv   data/val.csv \
  --image-root / \
  --device cuda
```

---

## 7. Evaluation Benchmarks

### WHO Target Product Profile (TPP) for triage

> **≥90% sensitivity at ≥70% specificity** is the minimum for a TB screening CAD product to be considered for WHO recommendation (June 2025 update).

Treat this as the **floor**, not the goal. WHO-recommended products achieve:

| Product | Sensitivity | Specificity | AUC |
| --- | --- | --- | --- |
| CAD4TB v6 | 90.0% | 73.8% | ~0.95 |
| qXR v3.0 | 90.0% | 75.0% | ~0.96 |
| Genki (DeepTek) | 90.0% | 71.2% | ~0.94 |

### Expected performance targets for this model

| Test set | AUC target | @ 90% sensitivity |
| --- | --- | --- |
| Shenzhen (internal) | ≥ 0.97 | spec ≥ 90% |
| Montgomery (external, low-burden US) | ≥ 0.92 | spec ≥ 75% |
| TBX11K (external, multi-national) | ≥ 0.90 | spec ≥ 70% |

### Minimum reporting requirements (TRIPOD+AI compliant)

Every evaluation report must include:

- [ ] Sensitivity, specificity, PPV, NPV at the WHO operating threshold (with 95% bootstrap CIs)
- [ ] AUC-ROC and AUC-PR (PR is more honest for rare-disease settings)
- [ ] F1 and Matthews Correlation Coefficient (MCC)
- [ ] Expected Calibration Error (ECE) and Brier score
- [ ] Subgroup analysis: sex, age band, HIV status, view (PA/AP), scanner, site
- [ ] Failure-mode analysis: Grad-CAM review of all false negatives by a radiologist
- [ ] Head-to-head comparison with ≥3 radiologists on the same blinded test set

### Subgroup analysis template

```python
# Run this after training to check for disparate performance
subgroups = {
    'HIV-positive'    : lambda r: r.get('hiv_status') == 'positive',
    'HIV-negative'    : lambda r: r.get('hiv_status') == 'negative',
    'AP view'         : lambda r: r.get('view_position') == 'AP',
    'PA view'         : lambda r: r.get('view_position') == 'PA',
    'Age 0-14'        : lambda r: r.get('age_band') == '0-14',
    'Age 60+'         : lambda r: r.get('age_band') == '60+',
    'Shenzhen scanner': lambda r: r.get('site') == 'shenzhen',
    'Montgomery scanner': lambda r: r.get('site') == 'montgomery',
}
```

---

## 8. Known Issues & Pitfalls

### P1 — Do not use horizontal flip

Already enforced in `augmentation.py` (`horizontal_flip = False` in all presets). If you add a custom transform pipeline, verify that `RandomHorizontalFlip` is **not** included. It is the default in most torchvision pipelines and produces inverted anatomy.

### P2 — Per-site calibration is mandatory

The raw model score is not a calibrated probability. Before deployment, run:

```bash
python scripts/deploy.py calibrate \
  --onnx deploy/model.onnx \
  --cal-csv data/site_local.csv \
  --image-root /path/to/local/images \
  --output deploy/calibrator.json
```

50–200 locally labelled CXRs are sufficient. Without calibration, sensitivity and specificity guarantees do not hold.

### P3 — DICOM PhotometricInterpretation

Some DICOM CXRs are stored as `MONOCHROME1` (lung = bright, background = dark) rather than the standard `MONOCHROME2`. The `load_dicom()` function handles this automatically via the `PhotometricInterpretation` tag, but raw `cv2.imread()` on DICOM files does not. Always use `load_image()` from `data.preprocessing`.

### P4 — Patient-level data split (avoid leakage)

The Shenzhen dataset has some patients with multiple CXRs at different timepoints. Use the `patient_id` column (if available) to ensure no patient appears in both train and test. `stratified_split()` does **not** check this automatically — you must pre-filter by patient.

```python
# Example: patient-level split
from collections import defaultdict
patients = defaultdict(list)
for rec in all_records:
    patients[rec['patient_id']].append(rec)
# Then split by patient, not by record
```

### P5 — ViT positional embedding interpolation

The `Conv-stem ViT-S/16` branch interpolates positional embeddings when the spatial grid at the stem output differs from the grid the ViT was pretrained on. This happens every forward pass when the input resolution differs from the ViT's native resolution (384px). This interpolation is bilinear and adds ~5ms per forward pass. To avoid it during inference, ensure input is always 384×384 for the ViT branch.

### P6 — TBX11K weakly-labelled noise

TBX11K labels were assigned by clinical review of patient records, not by a radiologist reading the X-ray itself. ~10–15% label noise is expected. Use a noisy-label-robust loss (the `AsymmetricLoss` in `training/losses.py` is robust to label noise by design) and downweight TBX11K samples if clean Shenzhen/Montgomery data is available.

### P7 — WHO does not recommend CAD for children (<15 years)

WHO's June 2025 position statement explicitly excludes paediatric use. If your dataset contains patients under 15, either:

- Exclude them from training/evaluation entirely, or
- Build and validate a dedicated paediatric model with self-supervised ViT pretraining (zero-shot paediatric TB is feasible via DINO pretraining)

### P8 — Score drift after software update

Model updates must be followed by threshold re-calibration on local data. The `InferenceEngine` drift monitor (in `deployment/inference.py`) will warn when the rolling mean score shifts by >5 points — treat this as a re-calibration trigger.

---

## 9. Checklist Before Training

Complete every item before starting a training run.

### Data

- [ ] All datasets downloaded and unzipped
- [ ] Unified `train.csv`, `val.csv`, `test.csv` generated with correct schema
- [ ] `split` column is populated (`stratified_split()` was used)
- [ ] No patient appears in both train and test splits
- [ ] TB-positive prevalence in `train.csv` is between 10–50% (WeightedSampler handles the rest)
- [ ] At least 2 geographic sites represented in training data
- [ ] `findings_label` column populated where radiologist labels are available
- [ ] No lateral views in dataset (or they are tagged `view_position=LL/RL` for filtering)

### Model

- [ ] `cfg.cnn.pretrained` is set (`"imagenet"` for baseline; `"mocov3_cxr"` + `pretrained_ckpt` for best results)
- [ ] `cfg.vit.pretrained` is set similarly
- [ ] Dry-run forward pass succeeds (`model(dummy_224)` outputs `tb_logits`, `tb_prob`, `findings_logits`)
- [ ] Grad-CAM enabled and tested on a sample image

### Training

- [ ] `freeze_epochs` set (default 3; increase to 5 if val loss spikes at unfreeze)
- [ ] `output_dir` points to a writable path with ≥5 GB free
- [ ] Mixed precision enabled (`mixed_precision = True`) if GPU VRAM < 24 GB
- [ ] W&B project name set (or W&B disabled via `wandb_enabled = False`)
- [ ] Seed set and `torch.manual_seed(seed)` called

### Evaluation

- [ ] Separate **external** test set (different geographic source from training data) prepared
- [ ] WHO TPP operating point evaluated: ≥90% sensitivity at ≥70% specificity
- [ ] Bootstrap CIs (n=2000) computed for AUC-ROC
- [ ] Subgroup analysis planned for sex, age, HIV status, view, scanner

### Deployment

- [ ] Per-site calibration data (50–200 locally labelled CXRs) identified
- [ ] ONNX export tested
- [ ] Inference latency measured on target device (target: <1s on mid-range GPU)
- [ ] Referral note ("confirmatory Xpert MTB/RIF required") verified in `TBPrediction.referral_note`

---

## Appendix A — TB manifestation labels (Head 2)

The 6-class findings head (`findings_label`) uses this ordering:

| Index | Finding | ICD-10 | Radiological description |
| --- | --- | --- | --- |
| 0 | Cavitation | A15.0 | Thick-walled lucency, usually upper lobe; indicates active replication |
| 1 | Consolidation | A15.0 | Homogeneous opacification of an airspace; lobar or segmental |
| 2 | Pleural effusion | J90 | Opacity at lung base; blunting of costophrenic angle |
| 3 | Hilar lymphadenopathy | R59 | Bilateral hilar enlargement; more common in primary TB |
| 4 | Fibrosis / scarring | A16.2 | Linear / reticular opacity; volume loss; indicates healed/inactive TB |
| 5 | Nodule(s) | A15.0 | Rounded opacity ≥3mm; isolated or miliary pattern |

When `findings_label = "0,0,0,0,0,0"` (all zeros), the findings head loss term still fires but contributes zero signal — this is handled correctly by `AsymmetricLoss` (easy negatives are down-weighted, not excluded).

---

## Appendix B — Regulatory pathway summary

| Jurisdiction | Pathway | Timeline | Notes |
| --- | --- | --- | --- |
| USA | FDA 510(k) | ~142 days | Use CAD4TB or qXR as predicate device |
| USA | FDA De Novo | ~10–11 months | If no suitable predicate |
| EU | MDR Class IIa | 6–12 months | Software as Medical Device (SaMD) |
| India | CDSCO Class B | 3–6 months | Required for deployment in high-burden settings |
| Global | WHO PQ / FIND | 12–18 months | Independent FIND evaluation against WHO TPP required |

WHO PQ (pre-qualification) for CAD devices requires:

1. Prospective multi-country clinical trial (≥3 countries, ≥2 of which are high-burden TB countries)
2. AUC ≥ 0.90 on each country's data independently
3. ≥90% sensitivity at ≥70% specificity at the claimed operating threshold
4. Demonstrated performance stability across scanner manufacturers and X-ray voltages

---

*Last updated: cough-vision v0.1.0*
