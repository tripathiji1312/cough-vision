# AGENTS.md — Working in cough-vision

This file is for automated agents (Claude, Codex, etc.) and human contributors.
Read it before writing any code.

---

## Project purpose

`cough-vision` is a clinically deployable deep-learning pipeline for pulmonary
TB detection from chest X-rays. WHO TPP target: >=90% sensitivity at >=70%
specificity (triage use). It is intended as Software as a Medical Device (SaMD).

---

## Install

```bash
# All runtime deps (torch, timm, opencv, sklearn, onnx, etc.)
pip install -e .

# With dev tools (pytest, black, mypy, isort):
pip install -e ".[dev]"

# With W&B training logging:
pip install -e ".[dev,train]"
```

---

## Repository layout

```
src/           Python package root — add src/ to sys.path or use pip install -e .
  config.py    Single source of truth for all hyperparameters
  data/        Preprocessing, augmentation, dataset classes
  models/      TBEnsemble, backbones, fusion, heads, Grad-CAM, segmentation
  training/    Losses, scheduler, fine-tune loop, MoCo pretraining
  evaluation/  Metrics (AUC, WHO TPP), threshold calibration
  deployment/  ONNX export, inference engine
scripts/       CLI entry points (train.py, evaluate.py, deploy.py)
tests/         Pytest test suite
notebooks/     Kaggle training notebook
plan/          Research plan (read-only reference)
DATASETS.md    Dataset schema, download instructions, pitfalls
```

---

## Import convention

All scripts and notebooks use `sys.path.insert(0, "src/")` before importing
from `config`, `models`, `training`, etc.  The `conftest.py` at the repo root
does this automatically for pytest — do not add per-file sys.path hacks in test
files.

```python
# Correct pattern in scripts/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import get_config
```

Intra-package imports use relative imports:

```python
# Inside src/data/dataset.py
from .preprocessing import load_image
from .augmentation import get_train_transform
```

---

## Code style conventions

1. **`from __future__ import annotations`** at the top of every source file.
2. **Dataclass configs** (`@dataclass`) in `config.py` — no dicts of dicts.
3. **Lazy import guards** for every optional dep:

   ```python
   _TORCH_AVAILABLE = False
   try:
       import torch as _t
       _TORCH_AVAILABLE = True
   except ImportError:
       _t = None
   def _require(flag, pkg): ...
   ```

4. **`Any` type hints** for uninstalled-dep types (do not use TYPE_CHECKING
   blocks for packages that may not be installed).
5. **No horizontal flip** in any augmentation path — placing the heart on the
   right side is anatomically impossible and was confirmed harmful in the
   2026 multi-national validation study.
6. **`torch.load` must always pass `weights_only=True`** for src/ code loading
   unknown checkpoints.  Use `weights_only=False` only in scripts/ with a
   `# nosec` comment and a reason.

---

## Commands

```bash
# Run all tests (skips if torch/cv2/timm not installed)
pytest tests/ -v

# Type check
mypy src/ --ignore-missing-imports

# Format
black src/ tests/ scripts/
isort src/ tests/ scripts/

# Lint
flake8 src/ tests/ scripts/ --max-line-length 100
```

---

## Key design decisions (do not change without discussion)

| Decision | Reason |
| --- | --- |
| No horizontal flip | Anatomically impossible; confirmed harmful in domain-shift study |
| Per-site threshold calibration | CAD4TB requires site-specific thresholds; skipping is a safety hazard |
| Lung-field masking | Suppresses scanner shortcuts (text, jewellery, pacemakers) |
| Multi-task findings head | Forces backbone to learn pathology not shortcuts |
| WHO TPP as primary metric | >=90% sens / >=70% spec, not just AUC |
| `scale_to_cad4tb` is LINEAR | NOT the real CAD4TB non-linear mapping; see heads.py docstring |

---

## Active bugs / known issues (check before editing)

See `/plans/audit-report-deep-dive.md` for the full audit. Currently resolved:

- CORRECT-01 (ViT stem stride for non-power-of-2 patch sizes)
- CORRECT-02 (WHO threshold selection with tied scores)
- DEBT-01 (.gitignore, .pyc files)
- DEBT-03 (dead get_parameter_groups in cnn.py)
- DEBT-04 (id()-set membership in ensemble get_parameter_groups)
- DEBT-05 (scale_to_cad4tb docstring)
- MIGR-01 (unused deps pruned from requirements.txt)
- SECURITY (weights_only=True added to torch.load)

Open direction items: DIR-01 (Grad-CAM wiring into deploy), DIR-02 (MoCo
pretraining path commitment), DIR-03 (dataset-prep script).
