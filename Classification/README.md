# Classification — CWA

One codebase for all classification datasets. `--dataset` selects everything
dataset-specific: loader, split, number of classes, image size, normalization,
augmentation policy, and hyperparameters.

## Install

```bash
pip install -r requirements.txt
```

`datasets` (Hugging Face) is only needed for Tiny ImageNet.

## Datasets

| `--dataset` | Source | Split | Classes |
|---|---|---|---|
| `burmese`, `potato`, `tomato` | ImageFolder tree on disk | per-class 70 / 15 / 15, seeded | from directories |
| `cifar100` | `torchvision.datasets.CIFAR100` (`download=False`) | official train → 45,000 / 5,000 stratified; official test held out | 100 |
| `tinyimagenet` | Hugging Face `zh-plus/tiny-imagenet` | official train → 90,000 / 10,000 stratified; official `valid` held out | 200 |

Point the code at your data with `--data-root`, or edit the dataset's config in
`configs/`. Nothing is ever downloaded for CIFAR-100.

Check that every image decodes before a long run:

```bash
python train.py --dataset cifar100 --data-root /path/to/parent --check-dataset
```

## Methods

Both are selected on **validation loss only** — the test set is never used to
pick or rank checkpoints, and nothing chooses a "best K" from test results.

| Label in results | What it is |
|---|---|
| `Baseline` | The single checkpoint with the lowest validation loss (conventional early stopping). |
| `CWA (K=n)` | **Proposed.** Uniform element-wise average of the Top-K checkpoints ranked by validation loss, then BatchNorm recalibration. Reported for every K in `TOP_K_VALUES` (default 2, 3, 4, 5). |

## Run

```bash
python train.py --dataset cifar100 --model vit_base
```

That runs every seed in `Config.SEEDS` sequentially, each as its own process
with its own output folder. For a single seed:

```bash
python train.py --dataset cifar100 --model vit_base --seed 42
```

### Reproducing the main tables

CIFAR-100 and Tiny ImageNet use per-model hyperparameters. Run each model, then
aggregate mean ± std over seeds from the per-seed Excel files.

```bash
python train.py --dataset cifar100 --model vit_base        --run-name vit_base        --batch-size 64 --epochs 50  --lr 2e-5   --weight-decay 1e-7 --warmup-epochs 3 --eta-min 1e-6 --early-stopping 10 --fc-layers 256 128 --dropout 0.5 --num-workers 8
```

```bash
python train.py --dataset cifar100 --model vgg16           --run-name vgg16           --batch-size 32 --epochs 70  --lr 5e-4   --weight-decay 1e-6 --warmup-epochs 7 --eta-min 1e-5 --early-stopping 10 --fc-layers 256 128 --dropout 0.5 --num-workers 8
```

```bash
python train.py --dataset cifar100 --model resnet101       --run-name resnet101       --batch-size 64 --epochs 50  --lr 6e-4   --weight-decay 1e-6 --warmup-epochs 5 --eta-min 1e-5 --early-stopping 10 --fc-layers 256 128 --dropout 0.6 --num-workers 8
```

```bash
python train.py --dataset cifar100 --model efficientnet_b0 --run-name efficientnet_b0 --batch-size 64 --epochs 50  --lr 6e-4   --weight-decay 1e-6 --warmup-epochs 5 --eta-min 1e-5 --early-stopping 10 --fc-layers 256 128 --dropout 0.5 --num-workers 8
```

```bash
python train.py --dataset cifar100 --model densenet121     --run-name densenet121     --batch-size 32 --epochs 50  --lr 5e-4   --weight-decay 1e-6 --warmup-epochs 5 --eta-min 1e-5 --early-stopping 10 --fc-layers 256 128 --dropout 0.5 --num-workers 8
```

```bash
python train.py --dataset cifar100 --model mobilenet_v2    --run-name mobilenet_v2    --batch-size 32 --epochs 150 --lr 1.5e-4 --weight-decay 1e-4 --warmup-epochs 9 --eta-min 1e-5 --early-stopping 15 --fc-layers 512     --dropout 0.4 --num-workers 8
```

Swap `--dataset cifar100` for `--dataset tinyimagenet` — the same per-model
settings were used for both.

The agricultural datasets use their config defaults, so no overrides are needed:

```bash
python train.py --dataset burmese --data-root /path/to/BurmeseGrapeDataset
```

## Output

Each run writes to `results/<run-name-or-number>/`:

```
run_config.json / run_config.xlsx    every setting that produced this run
all_models_results.xlsx              Overall Metrics + Per-Class Metrics sheets
performance_comparison.png           Baseline vs CWA across models and metrics
<model_name>/
  <model_name>_results.xlsx
  confusion_matrices/
  training_curves/                   loss/accuracy curves + history CSV
```

The `Method` column holds `Baseline` or `CWA (K=n)`. Training checkpoints are
deleted after evaluation unless `--keep-checkpoints` is passed.

## Layout

```
train.py         entrypoint: CLI, run folders, multi-seed dispatch
configs/         per-dataset config classes + the Config registry
data/            one loader per dataset, shared transforms and DataLoaders
models/          backbones + classifier heads
methods/         Baseline and CWA, plus the averaging/BN primitives
training/        training loop, checkpoint manager, optimizer, losses
evaluation/      metric computation and result export
utils/           run folders, seeding, plots, AMP context
```

### Adding a dataset

1. Add a loader module in `data/` exposing `load_splits`, `build_dataset`, `verify`.
2. Add a config class in `configs/` and register it in `configs/__init__.py`.

Nothing else changes — `--dataset yourname` works from there.

## Notes on reproducing exactly

Some settings differ **deliberately** between datasets because that is what the
reported runs used. They are set in config, not forked into separate code paths:

- **Optimizer parameter groups.** CIFAR-100 / Tiny ImageNet use timm's grouping
  (biases and 1D/norm parameters undecayed); the agricultural runs decay every
  trainable parameter. With `WEIGHT_DECAY=0.1` these are not equivalent.
- **AMP, Mixup/CutMix, gradient clipping** are on for CIFAR-100 / Tiny ImageNet
  and off for the agricultural datasets.
- **Augmentation policy** — `pretrain_224` vs `agri` (see `data/transforms.py`).
- **`ETA_MIN` > `LEARNING_RATE` on the agricultural datasets** (1e-5 vs 9e-7).
  The cosine phase therefore raises the learning rate instead of annealing it.
  This is what the reported runs did; it is preserved on purpose and the
  validator prints a note rather than an error.

## Optional: cross-validation

`--cv` runs stratified K-Fold over the training pool with the test set held out,
writing `cv_summary.xlsx` with mean ± std per method. The reported tables use
multi-seed runs, not CV; this is an additional option.
