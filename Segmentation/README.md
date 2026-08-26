# CWA — Instance Segmentation (Ultralytics YOLO-seg + Carparts)

This pipeline uses YOLO segmentation models (`-seg`) through the Ultralytics
API. All main configuration lives in `config.py`; `main.py` provides a CLI to
train, evaluate Baseline/CWA, export Excel and export models.

See [`FLOW.md`](FLOW.md) for the end-to-end execution flow.

## Install and run

```bash
pip install -r requirements.txt
```

The Ultralytics version is pinned in `requirements.txt` because the pipeline
relies on internal callbacks, the checkpoint format and `SegmentMetrics`.

Set `MODEL`, `DATA` and hyperparameters in `config.py`, or override on the CLI:

```bash
python main.py train --exp_name carparts_yolov8s_raw_topk_run01
```

```bash
python main.py methods --run-dir results/segmentation/<experiment>/seeds/seed_42
```

```bash
python main.py eval --weights <run>/weights/best.pt --split test
```

```bash
python main.py export --run-dir <run> --output segmentation_results.xlsx
```

```bash
python main.py export-model --weights <run>/weights/best.pt --format onnx
```

The pipeline rejects plain detection models such as `yolov8s.pt` — use a
segmentation model like `yolov8s-seg.pt`.

`--exp-name` and `--exp_name` are equivalent. With this argument the output is
`results/segmentation/<exp_name>/` and no timestamp is appended. A name that
already exists and is non-empty is rejected, so two experiments cannot be mixed.

## Tests

No dataset or trained model needed — the reporting layer is driven with
synthetic metrics:

```bash
python tests/test_reporting.py
```

## Reported methods

| Method | Description |
|---|---|
| `Baseline (best.pt)` | The temporary `best.pt` — the raw checkpoint with the highest raw-model validation fitness (FP16, via Ultralytics' `strip_optimizer`) |
| `Top-1 (best raw ckpt)` | The same checkpoint but **raw FP32, no averaging, BN untouched** — a K=1 baseline at the same precision as CWA. Toggle with `EVAL_TOP1_BASELINE` |
| `CWA (Top-K avg)` | Uniform element-wise average of the raw weights of the Top-K raw checkpoints + BN recalibration |

`Top-1` exists to separate two sources of difference: checkpoint precision
(FP16 vs FP32) and the real effect of averaging. It is also the K = 1 point of
the mAP-vs-K curve in `summary/charts/03_topk_curve_mAP50-95.png`.

Ultralytics' segmentation fitness is used consistently for `best.pt`, early
stopping and Top-K ranking:

```text
fitness =
    0.1 x mAP50(B) + 0.9 x mAP50-95(B)
  + 0.1 x mAP50(M) + 0.9 x mAP50-95(M)
```

`B` is bounding box, `M` is mask. Precision and Recall do not contribute to
fitness.

For the selected checkpoint set `S_K`, CWA computes:

```text
w_avg = (1 / K) x Σ w_t^raw,  t ∈ S_K
```

This is uniform element-wise averaging — no new parameter is learned during
averaging.

## Data split

The default is Ultralytics' `carparts-seg.yaml`:

| Split | Images | Purpose |
|---|---:|---|
| `train` | 3156 | Training and BN recalibration |
| `val` | 401 | Fitness, `best.pt`, early stopping, Top-K ranking |
| `test` | 276 | Final reporting |

`VAL_RATIO = 0` keeps those three independent splits as-is. For a custom
dataset you must ensure `val` does not overlap `test`. To carve validation out
of train instead, set `VAL_RATIO > 0` and `dataset.py` will create a
seed-deterministic split.

The pipeline uses the `carparts-seg.yaml` pinned in this project rather than the
older YAML bundled with `ultralytics==8.3.152`.

## Raw checkpoint averaging

Ultralytics maintains an EMA by default, validates with the EMA, and saves only
the EMA into checkpoints. This pipeline deliberately disables EMA at the start
of training so the method matches the raw checkpoint averaging in the paper:

1. The optimizer updates the raw `trainer.model`.
2. At the end of each epoch the raw `state_dict` is copied 1:1 into a separate
   validation module; validation and fitness use exactly that raw snapshot.
3. Early stopping and `best.pt` use raw-model fitness.
4. Each `epochN.pt` stores the raw FP32 model in the `model` field, with
   `ema=None`.
5. After training, the Top-K raw checkpoints are selected and their raw
   parameters uniformly averaged in FP32; the temporary averaged checkpoint is
   also FP32.
6. Once metrics/Excel are collected, all temporary checkpoints are deleted.

The separate validation module exists only to isolate `torch.inference_mode()`
from the training model; it performs no EMA smoothing. So there is no EMA
smoothing before Top-K averaging.

## Checkpoints and BatchNorm

When `USE_CWA=True`:

- `save_period=1` writes one `epochN.pt` per epoch.
- A callback disables EMA and reads the raw-model `trainer.fitness`.
- Only the `KEEP_TOP_K_CHECKPOINTS` highest-fitness checkpoints are kept.
- The ranking is written to `weights/cwa_checkpoints.json`.
- Raw FP32 **learnable parameters** (conv/linear weights, biases, BN γ/β) are
  uniformly averaged; checkpoints must share an architecture or
  `average_checkpoints` raises immediately.
- BN `running_mean`, `running_var`, `num_batches_tracked` are **not** averaged
  (they are population statistics, not learned parameters) — they are carried
  over from the rank-1 checkpoint, then reset and re-estimated by
  `update_bn_stats()`.
- Each `cwa_topK_avg.pt` is deleted as soon as `model.val()` finishes.
- At the end of each seed the whole `weights/` directory (`best.pt`, `last.pt`,
  raw `epochN.pt`, any leftover ranking JSON) is deleted in a `finally` block.

Old runs that used EMA, or an old ranking JSON, are rejected; they must be
retrained because the corresponding raw checkpoints do not exist.

### BN recalibration (`update_bn_stats`)

After averaging, the weights feeding each BN layer have changed but
`running_mean` / `running_var` still belong to the old checkpoint, so activation
statistics are stale. The procedure matches `torch.optim.swa_utils.update_bn`:

1. `reset_running_stats()` on every `_BatchNorm`, with `momentum = None`
   (cumulative moving average, not EMA).
2. The whole model in `eval()`, with only BN modules in `train()` so they
   accumulate statistics.
3. Iterate `BN_UPDATE_BATCHES` batches of the **train split** inside
   `torch.no_grad()` — **no** backward, **no** optimizer step, forward only.
4. Restore `momentum` and overwrite the checkpoint (only BN buffers changed;
   the learnable parameters keep the FP32 averaged values).

`BN_UPDATE_CLOSE_MOSAIC` controls the augmentation of the dataloader used to
estimate BN. Ultralytics turns off mosaic/mixup/cutmix/copy_paste for the final
`close_mosaic` epochs (default 10); estimating BN on mosaic images while the
checkpoint was trained in the non-mosaic phase would skew the statistics. The
default `"auto"` follows the rank-1 checkpoint's epoch: inside the last
`close_mosaic` epochs → mosaic off; stopped earlier → full augmentation.

The BN dataloader is released explicitly after use: Ultralytics'
`build_dataloader` returns an `InfiniteDataLoader` that keeps worker processes
alive, so without an explicit `del` every K value and every seed would add
another `WORKERS` processes.

## Ultralytics defaults left untouched

- **Loss**: the stock `v8SegmentationLoss` (box + mask + cls BCE + DFL) with
  `LOSS_FUNCTION = "bce"`. The `"focal"` option exists only for ablation.
- **SGD**: `momentum = 0.937`, `weight_decay = 5e-4` (Ultralytics scales it by
  batch), `warmup_momentum = 0.8`, `warmup_bias_lr = 0.1`, `nbs = 64`.
  `build_train_args()` does not pass these keys, so the trainer uses its
  defaults.
- Only these are overridden: `optimizer`, `lr0`, `lrf`, `warmup_epochs`,
  `cos_lr`, `mixup`, `copy_paste`.

## Dataloader workers

Ultralytics uses `workers x 2` for the validation dataloader during training
(`models/yolo/detect/train.py`), so `WORKERS = 16` means 32 validation workers
and frequent `DataLoader worker (pid ...) is killed by signal` errors or an
exhausted `/dev/shm`. `CAP_VAL_WORKERS = True` (default) forces the validation
loader to use exactly `WORKERS`.

## AMP

`AMP = True` applies to both training (Ultralytics' `amp`) and the forward pass
of BN recalibration. On H100, autocast selects `bfloat16`.

## Output

`python main.py train --exp-name yolov8s_exp1` creates exactly one directory
named after the experiment, containing only Excel, charts and logs — **no `.pt`
checkpoints**:

```text
results/segmentation/yolov8s_exp1/
├── README.md                          # quick mean ± std table, no need to open Excel
├── experiment_config.json             # full config snapshot for the run
├── summary/
│   ├── yolov8s_exp1_summary.xlsx      # aggregated results over 5 seeds
│   └── charts/
│       ├── 01_mAP50-95_by_strategy.png   # bar, mean ± std per method
│       ├── 02_mAP50_by_strategy.png
│       ├── 03_topk_curve_mAP50-95.png    # mAP vs K + baseline line
│       ├── 04_per_seed_mAP50-95.png      # one line per seed
│       ├── 05_delta_vs_baseline.png      # seed-paired Δ + win rate
│       └── 06_per_class_delta.png        # per-class Δ AP
└── seeds/
    ├── seed_1/
    │   ├── seed_1_results.xlsx        # Summary | PerEpoch | Checkpoints
    │   ├── charts/
    │   │   ├── training_curves.png        # Ultralytics' results.png
    │   │   ├── checkpoint_selection.png   # fitness per epoch with Top-K marked
    │   │   ├── confusion_matrix*.png
    │   │   └── Box*_curve.png, Mask*_curve.png
    │   └── logs/{results.csv, args.yaml}
    ├── seed_10/ ...
    └── seed_500/
```

### `summary/<exp_name>_summary.xlsx`

| Sheet | Content |
|---|---|
| `MeanStd` | **mean ± std across all seeds per method** — the main table for the paper (`std` is the sample std, `ddof=1`) |
| `DeltaVsBaseline` | **Seed-paired** Δ vs Baseline + how many seeds that method wins |
| `PerSeed` | One row per (seed × method) |
| `PerClass_MeanStd` | mean ± std AP per class × method |
| `PerClass_PerSeed` | Raw per-class AP |
| `Checkpoints` | Which epochs entered the Top-K per seed, with their fitness |
| `RunInfo` | Experiment config, checkpoint-selection criterion, failed seeds (if any) |

### `seeds/seed_<N>/seed_<N>_results.xlsx`

- `Summary`: run info + overall metrics per method (with a `Δ ... vs baseline`
  column) + mask AP per class per method.
- `PerEpoch`: every train/validation column in `results.csv` (box and mask).
- `Checkpoints`: the Top-K ranking by `val'` fitness.

By default no `weights/` directory survives a completed seed
(`DELETE_CHECKPOINTS_AFTER_RUN=True`). Checkpoints exist only temporarily during
ranking, averaging, BN update, evaluation and optional export. To keep or deploy
a `.pt` model, change this policy before training.

The sample images Ultralytics dumps (`train_batch*.jpg`, `val_batch*.jpg`,
`labels*.jpg`) are deleted by default (`KEEP_SAMPLE_IMAGES=False`) — they are
large and are not charts. Per-method `model.val()` plots are off by default
(`SAVE_EVAL_PLOTS=False`); when enabled they land in
`seeds/seed_<N>/eval_plots/` rather than Ultralytics' default `runs/segment/valN`.

A failing seed does not break the experiment: the pipeline records the error,
continues with the remaining seeds, and lists the failed ones in `README.md` and
the `RunInfo` sheet.

## Key configuration

| Group | Settings |
|---|---|
| Model/data | `MODEL`, `DATA`, `VAL_RATIO` |
| Training | `EPOCHS`, `IMGSZ`, `BATCH`, `DEVICE`, `PATIENCE`, `RANDOM_SEED` |
| Optimizer/LR | `OPTIMIZER`, `LR0`, `LRF`, `WARMUP_EPOCHS`, `COS_LR` |
| Augmentation | `MIXUP`, `COPY_PASTE`, `EXTRA_TRAIN_ARGS` |
| CWA | `TOP_K_VALUES`, `KEEP_TOP_K_CHECKPOINTS`, `EVAL_TOP1_BASELINE` |
| BN recalibration | `USE_BN_UPDATE`, `BN_UPDATE_BATCHES`, `BN_UPDATE_CLOSE_MOSAIC`, `AMP` |
| Evaluation | `EVAL_SPLIT`, `CONF`, `IOU` |
| Output | `PROJECT`, `EXP_NAME`, `DELETE_CHECKPOINTS_AFTER_RUN`, `KEEP_SAMPLE_IMAGES`, `SAVE_EVAL_PLOTS`, `MAKE_CHARTS` |

`PATIENCE` counts epochs without a strict fitness improvement; early stopping is
not based on validation loss.

`RANDOM_SEED` accepts a list (default `[1, 10, 42, 100, 500]`); each seed is an
independent run under `seeds/seed_<N>/` and all are aggregated into `summary/`.
