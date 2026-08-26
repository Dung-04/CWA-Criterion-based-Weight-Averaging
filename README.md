# CWA — Checkpoint Weight Averaging

Code release for the CWA paper. CWA averages the Top-K checkpoints ranked by
validation performance and recalibrates BatchNorm, instead of shipping the
single best checkpoint.

Three tasks, three self-contained codebases:

| Directory | Task | Dataset(s) | Framework |
|---|---|---|---|
| [`classification/`](classification/) | Image classification | Burmese grape, Potato, Tomato, CIFAR-100, Tiny ImageNet | timm / torchvision |
| [`detection/`](detection/) | Object detection | Pascal VOC | Ultralytics YOLO |
| [`segmentation/`](segmentation/) | Instance segmentation | Carparts | Ultralytics YOLO-seg |

Each has its own `README.md` and `requirements.txt`; install per task.

## Methods

Every task reports the same two methods under the same names:

| Name | Description |
|---|---|
| **Baseline** | The single best checkpoint on validation (conventional early stopping). |
| **CWA** | *Proposed.* Uniform element-wise average of the Top-K validation-ranked checkpoints, followed by BatchNorm recalibration. Reported for K = 2, 3, 4, 5. |

Detection and segmentation additionally report two controls that isolate where
the gain comes from:

- **Baseline + BN recal** (detection) — the baseline checkpoint with BatchNorm
  recalibrated but no averaging, so the improvement cannot be attributed to BN
  recalibration alone.
- **Top-1 (best raw ckpt)** (segmentation) — the same checkpoint in raw FP32
  with no averaging, so FP16-vs-FP32 precision is not mistaken for an averaging
  effect. It is also the K = 1 point of the mAP-vs-K curve.

**No test-set leakage.** Checkpoints are selected and ranked purely on a
validation split that is disjoint from the test set, and no code path picks a
"best K" from test results — every K is reported.

## Quick start

```bash
cd classification && pip install -r requirements.txt
```

```bash
python train.py --dataset cifar100 --model vit_base --seed 42
```

```bash
cd detection && pip install -r requirements.txt
```

```bash
python main.py train --exp-name voc_yolov8s_run01
```

```bash
cd segmentation && pip install -r requirements.txt
```

```bash
python main.py train --exp-name carparts_yolov8s_run01
```

See each task's README for dataset setup, the exact per-model hyperparameters,
and how to reproduce the reported tables.

## Reproducibility

- **Seeds.** Classification, detection and segmentation all run seeds
  `1, 10, 42, 100, 500` and report mean ± std across them. The agricultural
  classification datasets use seed `42`.
- **Splits.** Detection splits an independent `val'` out of VOC train, because
  stock `VOC.yaml` makes `val` and `test` the same images — selecting
  checkpoints on that would leak the test set. Segmentation's Carparts YAML
  already has three disjoint splits.
- **Determinism.** Classification accepts `--deterministic`; the Ultralytics
  pipelines set `deterministic=True` by default.

## Repository layout

```
classification/
  train.py       configs/   data/     models/
  methods/       training/  evaluation/  utils/
detection/
  main.py        train.py   evaluate.py  reporting.py
  config.py      dataset.py losses.py    memory.py
segmentation/
  main.py        train.py   evaluate.py  charts.py
  config.py      dataset.py losses.py    carparts-seg.yaml
```

Detection and segmentation stay flat: each is a single task on a single dataset,
so there is nothing to abstract over.
