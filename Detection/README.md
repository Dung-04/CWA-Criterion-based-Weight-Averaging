# CWA — Object Detection (Ultralytics YOLO + Pascal VOC)

Trains an object detector with Ultralytics, selects the Top-K checkpoints by
validation fitness, and uniformly averages their raw learnable parameters.
The library version is pinned at `ultralytics==8.3.152`.

## Install and run

```bash
pip install -r requirements.txt
```

```bash
python main.py train --exp-name yolov8s_exp1
```

`--exp-name` and `--exp_name` are both accepted. One command runs all five
seeds in `RANDOM_SEED` and produces:

```text
results/detection/yolov8s_exp1/
├── SUMMARY.xlsx              ★ mean ± std over all 5 seeds x every method
├── experiment_config.json    config snapshot for this run
├── charts/
│   ├── 01_topk_curve_mAP50-95.png   mAP vs K, ±std band, baseline level
│   ├── 02_method_mAP50-95.png       dot plot, mean ± std, every method
│   ├── 03_method_mAP50.png
│   ├── 04_delta_vs_baseline.png     Δ vs baseline with per-seed dots
│   └── 05_per_seed_paired.png       one line per seed (paired)
└── seeds/
    ├── seed_1/
    │   ├── results_seed_1.xlsx
    │   └── charts/
    │       ├── training/            curves + confusion matrix on val'
    │       ├── test_best/           PR/F1 curve + confusion matrix on test
    │       ├── test_best_bn/
    │       └── test_top_2/ … test_top_5/
    └── seed_10/ …
```

No `.pt` files are kept. An experiment name that already exists and is
non-empty is rejected, so two runs cannot be mixed.

A seed that fails (OOM, corrupt data) does **not** lose the seeds that already
finished: the error is recorded, the next seed runs, and `SUMMARY.xlsx` is
still exported from the successful seeds with the failed ones listed at the end
of the log.

Other commands:

```bash
python main.py train --exp-name baseline_run01 --no-cwa
```

```bash
python main.py train --exp-name k_sweep01 --top-k 2 3 5 8
```

```bash
python main.py train --exp-name no_bn01 --no-bn-update
```

```bash
python main.py train --exp-name debug01 --keep-checkpoints
```

```bash
python main.py eval --weights path/to/model.pt --split test
```

```bash
python main.py methods --run-dir results/detection/yolov8s_exp1/seeds/seed_42
```

## Tests

No dataset or trained model needed — the reporting layer is driven with
synthetic metrics:

```bash
python tests/test_reporting.py
```

## Data split and test-leakage prevention

Ultralytics' stock `VOC.yaml` declares:

| Split | Images | Source |
|---|---|---|
| train | 16,551 | VOC2007 trainval (5,011) + VOC2012 trainval (11,540) |
| val | 4,952 | VOC2007 test |
| test | 4,952 | **the same VOC2007 test images** — no separate holdout |

So `val ≡ test`. Because CWA selects checkpoints on validation, using this
as-is would select checkpoints on the very set being reported — leakage.
`dataset.py` therefore carves an independent validation split out of train with
`VAL_RATIO=0.1`:

| Split | Images | Role |
|---|---|---|
| `train'` | 14,896 | Train the model; re-estimate BatchNorm after averaging |
| `val'` | 1,655 | Fitness, early stopping, best checkpoint, Top-K ranking |
| `test` | 4,952 | VOC2007 test, untouched; final reporting only |

The split uses the run's own seed (`random.Random(seed).shuffle`), so each seed
gets a different `train'/val'` split — split variance is therefore included in
the std across seeds. The split file is cached per `(seed, ratio)` at the
dataset root (`holdout_seed<S>_val10.{yaml,txt}`), so re-running the same seed
reproduces it exactly.

**K is never chosen by looking at test results.** The code prints every K in
`TOP_K_VALUES`; no line selects a "best method" from test.

## Raw checkpoints and the averaging step

With CWA enabled:

1. Ultralytics' EMA smoothing is disabled.
2. After each epoch the raw training weights are copied exactly into a separate
   validation module. Validation, fitness, early stopping and `best.pt` all
   follow the raw weights.
3. A callback saves the raw model in FP32 to `epochN.pt`, tags it
   `weight_source="raw"`, and keeps only the `KEEP_TOP_K_CHECKPOINTS`
   checkpoints with the highest validation fitness.
4. For each predetermined K the code computes

   **w_avg = (1/K) · Σ_{t ∈ S_K} w_t**

   a uniform element-wise average of all learnable parameters: convolution and
   linear weights, biases, and BatchNorm affine parameters (γ, β).
   **This is not EMA**: there is no decay factor, no order dependence, and
   every checkpoint contributes exactly `1/K` regardless of its rank or epoch.
5. Non-learnable state is not averaged — `running_mean`, `running_var`,
   `num_batches_tracked`, anchors, stride, caches. These are not part of **w**;
   they are carried over from the best checkpoint and then re-estimated (see
   below).
6. The average and the resulting checkpoint stay FP32; summation is done in
   float64 before casting back to the original dtype, so the result does not
   depend on summation order and rounding error does not accumulate with K.
7. Before averaging, the code verifies that every checkpoint has the same key
   set and the same shapes; a mismatch stops the run rather than being silently
   skipped.

## BatchNorm recalibration

After averaging, the weights feeding each BN layer have changed but
`running_mean` / `running_var` still belong to the old checkpoint, so the
activation statistics are stale. `evaluate.update_bn_stats()` re-estimates them
(equivalent to `torch.optim.swa_utils.update_bn`):

1. `reset_running_stats()` and `momentum = None` → BN uses a **cumulative
   moving average**, i.e. the exact mean over every batch seen, independent of
   batch order and with no momentum.
2. **A forward-only pass over training data**:
   - everything inside `torch.no_grad()` — no graph, no backward;
   - no optimizer is created, no `optimizer.step()`;
   - `model.requires_grad_(False)` across the network;
   - `model.eval()` across the network, with only BN modules set to `.train()`
     so they accumulate mean/var;
   - after the loop the code **asserts** that every `parameter.grad is None`
     and raises otherwise — a runnable proof that no gradient update occurred.
3. Only `running_mean` / `running_var` / `num_batches_tracked` change; the
   learnable parameters keep exactly the averaged values.

The data used is the **`train'` split** — `val'` and `test` are never touched.
The dataloader runs in `mode='train'` (`BN_UPDATE_AUGMENT=True`) so the image
distribution matches the one the original BN statistics were accumulated over.

Because CWA is BN-recalibrated and the baseline is not, `BN_UPDATE_CONTROL=True`
adds a **"Baseline + BN recal"** row — the baseline checkpoint with BN
recalibrated — separating the gain from *averaging* from the gain from *BN
recalibration*. Without this control the improvement cannot be attributed to
the method.

Detection fitness in Ultralytics 8.3.152 is:

**fitness = 0.1 · mAP@0.5 + 0.9 · mAP@0.5:0.95**

`trainer.fitness` is exactly the criterion used to rank checkpoints and to
drive early stopping.

## Classification loss (`LOSS_FUNCTION`)

`'bce'` keeps the Ultralytics default; `'focal'` replaces `criterion.bce` with
`FocalBCE` (BCE × `(1-p_t)^γ` × α-factor, element-wise shape preserved).

Installation detail: patching `yolo_model.model` before training does **not**
work. `Model.train()` builds a completely new `DetectionModel`:

```python
self.trainer.model = self.trainer.get_model(weights=..., cfg=self.model.yaml)
```

so anything attached to the old module is discarded and training silently falls
back to BCE. The loss is therefore installed in the `on_train_start` callback
(when the model is already on the right device — `v8DetectionLoss` captures the
device at construction), for **both** `trainer.model` and `trainer.ema.ema`, so
the `val/cls_loss` column in `results.csv` is on the same scale as
`train/cls_loss`. The loss is assigned to `model.criterion` rather than
monkey-patching `init_criterion`, because binding a method onto an instance
breaks checkpoint pickling.

`assert_cls_loss_installed` runs immediately afterwards and **raises** if the
active loss is not `FocalBCE` — a future Ultralytics version that breaks this
mechanism is caught immediately instead of silently corrupting an experiment.

## Checkpoint lifecycle

Checkpoints are temporary artifacts:

- During training: at most the Top-K raw epoch checkpoints are kept, plus
  `best.pt`/`last.pt` which the trainer requires.
- Each averaged checkpoint is deleted as soon as `model.val()` finishes.
- After metrics/Excel are exported (and the deploy model, if enabled), the
  seed's entire `weights/` directory is deleted — even if evaluation failed.
- `tidy_run_dir()` then reduces the seed directory to **Excel + `charts/`**:
  images and plots move into `charts/training/`, `results.csv` and `args.yaml`
  are removed (their content is already in the `PerEpoch` and `RunInfo`
  sheets), and the `train_batch*`/`val_batch*` debug images are removed.
  Unrecognised files are **left alone** rather than blindly deleted.

Pass `--keep-checkpoints` to disable both steps for debugging. The
`methods`/`eval` commands on an old run only work if that run was trained with
`--keep-checkpoints` (or `DELETE_CHECKPOINTS_AFTER_RUN=False`).

## Key configuration

| Setting | Meaning |
|---|---|
| `MODEL` | Detection model, e.g. `yolov8s.pt` |
| `VAL_RATIO` | Fraction split off train as independent validation; `0.1` for VOC |
| `PATIENCE` | Early stopping on raw validation fitness |
| `RANDOM_SEED` | Int or list of seeds; a list runs them in turn and aggregates mean ± std |
| `TOP_K_VALUES` | The predetermined K values to report |
| `KEEP_TOP_K_CHECKPOINTS` | Must be ≥ `max(TOP_K_VALUES)` |
| `BASELINE_FROM_RAW_TOPK` | Take the baseline from rank #1 raw FP32 instead of FP16 `best.pt` |
| `USE_BN_UPDATE` | Re-estimate BN buffers on `train'` after averaging |
| `BN_UPDATE_BATCHES` | Number of forward batches for BN recalibration |
| `BN_UPDATE_AUGMENT` | Dataloader `mode='train'` (matches the training distribution) |
| `BN_UPDATE_CONTROL` | Add the "Baseline + BN recal" ablation row |
| `DELETE_CHECKPOINTS_AFTER_RUN` | Delete temporary checkpoints after each seed |
| `TIDY_RUN_DIR` | Reduce each seed directory to Excel + charts |
| `MAKE_CHARTS` | Render the experiment-level summary charts |
| `EVAL_SPLIT` | Split used for final reporting, default `test` |
| `EXP_NAME` | Stable experiment name on the server |
| `WORKERS` | Dataloader workers during training |
| `EVAL_WORKERS` | Workers for `model.val()` + BN recal (default 8) — see the RAM section |
| `AUTO_LIMIT_WORKERS` | Cap workers by the CPUs SLURM actually granted (default `True`) |

Why `BASELINE_FROM_RAW_TOPK=True`: Ultralytics saves `best.pt` through
`strip_optimizer()` in FP16, while the averaged checkpoint is FP32. Rank #1 in
the ranking table is the *same epoch with the same weights* as `best.pt` but
kept in FP32, so using it as the baseline puts both arms through the same
precision and the same load/eval path. The ranking tie-break (fitness
descending, earlier epoch on a tie) is set to match Ultralytics' `best_fitness`
semantics exactly, so rank #1 always lands on the same epoch as `best.pt`.

## Output metrics

`seeds/seed_<N>/results_seed_<N>.xlsx` — 5 sheets:

| Sheet | Content |
|---|---|
| `RunInfo` | All hyperparameters plus notes on the data split |
| `Overall` | One row per method: P, R, mAP@0.5, mAP@0.75, mAP@0.5:0.95, fitness, source epochs |
| `PerClass` | Per-class AP × method |
| `TopK_Checkpoints` | Which epochs entered the Top-K, their `val'` fitness, which K used them |
| `PerEpoch` | Ultralytics' `results.csv` verbatim |

`SUMMARY.xlsx` — 7 sheets:

| Sheet | Content |
|---|---|
| `Mean_Std` | ★ One row per method, with a ready `"0.5150 ± 0.0047"` column to paste into the paper |
| `PerSeed` | Raw numbers, one row per (seed × method) |
| `Delta_vs_Baseline` | Paired Δ vs baseline: mean ± std, seeds won, t-statistic, p-value |
| `PerClass_Mean_Std` | Per-class AP, mean ± std across seeds |
| `PerClass_PerSeed` | Per-class AP, raw |
| `TopK_Checkpoints` | Which epochs entered the Top-K per seed |
| `Config` | Config snapshot for the run |

`Delta_vs_Baseline` uses a **paired two-sided t-test**: the same seed means the
same data split and the same training run, so a paired comparison matches the
experimental design (an unpaired test would be swamped by between-seed
variance, which is several times larger than the effect). `p` is computed with
a regularized incomplete beta function, so scipy is not required.

## Host RAM (not VRAM)

Multiple seeds run in the **same process**, so RAM must return to its previous
level after each seed. Two things prevent that if left alone — both are handled
in `memory.py`:

1. Ultralytics' `InfiniteDataLoader` creates `self.iterator` in `__init__` and
   holds it for the object's lifetime, so `workers` child processes stay alive
   until the loader is destroyed, each holding `prefetch_factor` (=2) batches.
2. The `DetMetrics` returned by `model.val()` holds `on_plot`, a bound method of
   the validator, which in turn holds `validator.dataloader` from (1). Per-seed
   results are kept until the end of the experiment, so merely storing
   `DetMetrics` prevents every eval dataloader from ever being reclaimed.

With `USE_CWA=True`, **each** seed builds 3 dataloaders during training + 6
`model.val()` passes + 5 BN recalibrations = 14 dataloaders. At `WORKERS=24`
that is 336 child processes still alive after the first seed, and the second
seed dies while building the dataset (`Killed` / `slurmstepd: ... oom_kill`).

The pipeline therefore:

- returns a `MetricsSnapshot` (floats and strings only) instead of `DetMetrics`;
- closes dataloaders **explicitly** (`_shutdown_workers`) after each val pass,
  each BN recalibration and each seed, rather than relying on GC;
- uses `EVAL_WORKERS` (default 8) for everything after training: the prefetch
  queue costs `workers × 2 × batch × 3 × imgsz²` bytes — with
  `WORKERS=24, batch=128, imgsz=640` that is ~7.5 GB for **one** dataloader.
  Changing this value does not affect metrics, only speed and RAM.

Each seed logs `RSS ... | ... worker process` at the start. That number must
stay **flat** across seeds; if it climbs, a dataloader is still being retained.

### Why `WORKERS=16` shows up as 32 workers in the log

Two ×2 factors live inside Ultralytics 8.3.152:

- `DetectionTrainer.get_dataloader` (`models/yolo/detect/train.py:88`):
  `workers = self.args.workers if mode == "train" else self.args.workers * 2`
- `BaseTrainer._setup_train` (`engine/trainer.py:312`) calls it with `batch_size * 2`

So two loaders are always alive during training, and the validation loader costs
roughly 4× the training loader:

| Loader | Workers | Batch | Prefetch (imgsz 640) |
|---|---|---|---|
| `train_loader` | `WORKERS` | `BATCH` | 16 × 2 × 128 → 5.0 GB |
| `test_loader` | `WORKERS × 2` | `BATCH × 2` | 32 × 2 × 256 → 20.1 GB |

The warning `This DataLoader will create 32 worker processes in total` appears
because `build_dataloader` caps workers by `os.cpu_count()` — the **whole
node's** CPUs — while PyTorch compares against `os.sched_getaffinity`, the CPUs
**granted to the job**. On a 96-CPU node with a 16-CPU allocation, Ultralytics
still creates all 32.

`AUTO_LIMIT_WORKERS=True` (default) lowers `WORKERS` to `cpu_quota // 2` so no
loader exceeds the granted CPUs, and lowers `EVAL_WORKERS` to `cpu_quota`.
`validate_config()` prints the effective numbers plus a prefetch RAM estimate.

`CACHE` is **unrelated** to this problem: it already defaults to `False` (images
are not cached in RAM). Only `CACHE="ram"` loads the whole dataset into RAM —
do not enable that on VOC.

## Troubleshooting

- **Host OOM** (`Killed`, `slurmstepd: oom_kill` — not `CUDA out of memory`):
  lower `--eval-workers` first, then `--workers`, or request more `--mem` for
  the job. `validate_config()` prints a queue RAM estimate when it exceeds 8 GB.
- **VRAM OOM** (`CUDA out of memory`): lower `BATCH`/`IMGSZ`, or use `--batch -1`.
- **Not enough Top-K checkpoints**: the number of epochs actually run before
  early stopping is smaller than K.
- **An old run reports non-raw checkpoints**: it must be retrained with the
  current code; the pipeline deliberately refuses to mix EMA checkpoints into
  raw averaging.
