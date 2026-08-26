# Pipeline flow

Command: `python main.py train --exp-name yolov8s_exp1`

## Overview

```text
main.py  → Config.validate_config()  → train.train_detector()
                                          │
                                          ├── create results/segmentation/yolov8s_exp1/
                                          ├── for seed in [1, 10, 42, 100, 500]:   ← 5 independent runs
                                          │      A. TRAIN   (Ultralytics + 3 callbacks)
                                          │      B. EVAL    (Baseline / Top-1 / Top-K avg)
                                          │      C. EXCEL + CHARTS for the seed
                                          │      D. DELETE checkpoints + tidy the directory
                                          └── E. AGGREGATE 5 seeds → mean ± std + charts + README
```

## A. Training one seed

Model: `Config.MODEL` (yolov8s-seg.pt). Loss: **Ultralytics default**
(`v8SegmentationLoss` = box + mask + cls BCE + DFL). Optimizer: SGD, `lr0=5e-3`,
`lrf=0.01`, 5 warmup epochs, cosine LR; `momentum=0.937` and
`weight_decay=5e-4` are **left at the Ultralytics defaults**. AMP is on
(bfloat16 on H100).

Three callbacks are attached:

| Callback | When | What it does |
|---|---|---|
| `cap_val_dataloader_workers` | `on_pretrain_routine_start` | Force the val dataloader to use exactly `WORKERS` (Ultralytics uses `workers × 2`) |
| `TopKCheckpointManager.on_train_start` | before epoch 1 | `ema.enabled = False` → **disable EMA** |
| `.on_train_epoch_end` | after the epoch's last optimizer step | Copy the raw `state_dict` 1:1 into a separate validation module |
| `.on_model_save` | after Ultralytics writes `epochN.pt` | Overwrite it with the **raw FP32** model, record fitness, and **prune** anything outside the Top-5 |

Per-epoch loop (driven by Ultralytics):

```text
train epoch  →  sync raw weights into the validation model  →  validate on the `val` split
             →  fitness = 0.1·mAP50(B) + 0.9·mAP50-95(B) + 0.1·mAP50(M) + 0.9·mAP50-95(M)
             →  best.pt if fitness is the highest so far  →  early stopping (patience=10)
             →  save epochN.pt (raw FP32) → keep exactly the 5 highest-fitness files
```

At the end, disk holds `best.pt`, `last.pt`, five `epochN.pt` files and
`cwa_checkpoints.json` (the ranking table).

## B. Evaluation on `test` (`run_method_evaluation`)

`rank_checkpoints()` reads the JSON → a list of `(file, fitness, epoch)` in
descending order.

| # | Method | Weights | BN |
|---|---|---|---|
| 1 | `Baseline (best.pt)` | best.pt (FP16) | untouched |
| 2 | `Top-1 (best raw ckpt)` | `ranked[0]` raw FP32, **no averaging** | untouched |
| 3 | `CWA (Top-2 avg)` | average of the first 2 checkpoints | **recalibrated** |
| 4 | `CWA (Top-3 avg)` | average of the first 3 checkpoints | **recalibrated** |
| 5 | `CWA (Top-4 avg)` | average of the first 4 checkpoints | **recalibrated** |
| 6 | `CWA (Top-5 avg)` | average of the first 5 checkpoints | **recalibrated** |

Rows 3–6 each run exactly three steps:

```text
1) average_checkpoints(ranked[:K])
      w_avg = (1/K) · Σ w_t            ← uniform element-wise, NO EMA/decay
      applied to: conv/linear weights, biases, BN γ and β
      CARRIED OVER from the rank-1 ckpt: running_mean, running_var, num_batches_tracked

2) update_bn_stats(avg.pt, data)
      reset_running_stats() on every BN, momentum = None (cumulative average)
      model.eval(), with only BN modules in .train()
      iterate 100 batches of the TRAIN split inside torch.no_grad()   ← forward only,
      no backward, no optimizer step
      augmentation matched to the checkpoint's training phase (BN_UPDATE_CLOSE_MOSAIC="auto")

3) model.val(split="test")  →  SegmentMetrics (mask P, R, mAP50, mAP75, mAP50-95)
      → delete avg.pt immediately
```

## C–D. Per-seed results, then cleanup

`seed_<N>_results.xlsx` (Summary | PerEpoch | Checkpoints) and the
`checkpoint_selection.png` chart are written. Then the whole `weights/`
directory is deleted, sample images (`train_batch*`, `val_batch*`, `labels*`)
are removed, charts are collected into `charts/`, and `results.csv` + `args.yaml`
into `logs/`.

A failing seed does not stop the experiment — the error is recorded and the next
seed runs.

## E. Aggregating the 5 seeds

`summary/yolov8s_exp1_summary.xlsx`:

- `MeanStd` — **mean ± std (ddof=1) over the 5 seeds per method** ← the main table
- `DeltaVsBaseline` — seed-paired Δ vs Baseline + how many seeds won
- `PerSeed`, `PerClass_MeanStd`, `PerClass_PerSeed`, `Checkpoints`, `RunInfo`

`summary/charts/`: mean±std bars, **mAP vs K**, per-seed lines, paired Δ, and
per-class Δ. The `README.md` at the experiment root carries the mean ± std table
as markdown.

## Final directory

```text
results/segmentation/yolov8s_exp1/
├── README.md
├── experiment_config.json
├── summary/{yolov8s_exp1_summary.xlsx, charts/}
└── seeds/seed_<N>/{seed_<N>_results.xlsx, charts/, logs/}
```

No `.pt` file is retained.

## Data-control checkpoints

- Checkpoints are **selected** by fitness on the `val` split (401 images).
- BN is **recalibrated** on the `train` split (3156 images).
- Reported numbers come from the `test` split (276 images), which takes part in
  none of the steps above.
