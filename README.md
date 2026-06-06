# SCKATision-Net

PyTorch implementation of **SCKATision-Net**, a KAN-enhanced hybrid transformer network for microscopic white blood cell classification.

The implementation follows the manuscript design: white blood cell images are resized to `224×224×3`, normalized to `[0, 1]`, passed through an SC-Conv encoder, converted to patch tokens, refined with a Global-Local Attention Encoder, processed by transformer blocks where the standard MLP is replaced by a KAN-style feed-forward module, and classified into five WBC categories.

## Architecture

Main modules:

- **SC-Conv Encoder**: Spatial Reconstruction Unit (SRU) + Channel Reconstruction Unit (CRU) for spatial/channel redundancy reduction.
- **Patch Embedding**: Non-overlapping image patches projected to a token sequence.
- **GLAE**: Local depth-wise convolution plus multi-head self-attention for local and global context.
- **KAN Transformer Blocks**: Transformer encoder blocks where the feed-forward MLP is replaced by practical RBF KAN layers.
- **Classification Head**: Class-token representation mapped to WBC classes.

Default classes:

```text
basophil, eosinophil, lymphocyte, monocyte, neutrophil
```

## Repository layout

```text
sckatisation_net_repo/
├── configs/
│   ├── default.yaml
│   └── lisc_augmented.yaml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── gradcam.py
├── sckatisation_net/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── engine.py
│   ├── metrics.py
│   ├── model.py
│   └── utils.py
├── tests/
│   └── test_model.py
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <your-repo-url>
cd sckatisation_net_repo
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Install the project in editable mode for cleaner imports:

```bash
pip install -e .
```

A minimal `pyproject.toml` is included so editable installation works.

## Dataset format

Two formats are supported.

### Option A: one ImageFolder directory with automatic validation split

```text
data/Raabin-WBC/
├── basophil/
├── eosinophil/
├── lymphocyte/
├── monocyte/
└── neutrophil/
```

Run with `configs/default.yaml`. The script will create a reproducible train/validation split.

### Option B: explicit train/val/test folders

```text
data/Raabin-WBC/
├── train/
│   ├── basophil/
│   └── ...
├── val/
│   ├── basophil/
│   └── ...
└── test/
    ├── basophil/
    └── ...
```

Use `--split-folders` for this layout.

## Training

Edit `configs/default.yaml` so `data.data_dir` points to your dataset, then run:

```bash
python scripts/train.py --config configs/default.yaml
```

For explicit split folders:

```bash
python scripts/train.py --config configs/default.yaml --split-folders
```

For LISC-style small-data augmentation:

```bash
python scripts/train.py --config configs/lisc_augmented.yaml
```

Outputs are saved under `train.output_dir`, including:

- `best.pt` — best validation checkpoint
- `last.pt` — latest checkpoint
- `history.json` — epoch metrics
- `run_metadata.json` — config and class names

## Evaluation

Evaluate an ImageFolder test directory:

```bash
python scripts/evaluate.py \
  --checkpoint runs/sckatisation_net/best.pt \
  --data-dir data/Raabin-WBC-test \
  --output runs/sckatisation_net/test_metrics.json
```

Evaluate explicit `test/` split:

```bash
python scripts/evaluate.py \
  --checkpoint runs/sckatisation_net/best.pt \
  --data-dir data/Raabin-WBC \
  --split-folders \
  --output runs/sckatisation_net/test_metrics.json
```

## Prediction

```bash
python scripts/predict.py \
  --checkpoint runs/sckatisation_net/best.pt \
  --image sample_wbc.png
```

## Grad-CAM explainability

```bash
python scripts/gradcam.py \
  --checkpoint runs/sckatisation_net/best.pt \
  --image sample_wbc.png \
  --output gradcam.png
```

The Grad-CAM hook targets the final SC-Conv feature layer, producing a heatmap over image regions used for classification.

## Verification

Run the included smoke tests:

```bash
pytest -q
```

The tests verify the KAN layer and a small SCKATision-Net forward pass.

## Reproducibility notes

The manuscript reports training for 100 epochs with batch size 16, AdamW, learning rate `1e-4`, weight decay `1e-4`, and cosine restart scheduling. Those defaults are encoded in `configs/default.yaml`.

Exact published scores depend on access to the same train/validation/test splits and original datasets. This repository provides an end-to-end implementation and training pipeline, not the dataset files.

## Citation

Use the associated manuscript when citing this code:

```text
SCKATision-Net: A KAN-Enhanced Dual Encoder Transformer Hybrid Network with Spatial Channel Convolutional-Attention for Accurate Classification of Microscopic White Blood Cells.
```
