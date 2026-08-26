# Flow của pipeline

Lệnh: `python main.py train --exp-name yolov8s_exp1`

## Toàn cảnh

```text
main.py  → Config.validate_config()  → train.train_detector()
                                          │
                                          ├── tạo results/segmentation/yolov8s_exp1/
                                          ├── for seed in [1, 10, 42, 100, 500]:   ← 5 run độc lập
                                          │      A. TRAIN   (Ultralytics + 3 callback)
                                          │      B. EVAL    (Baseline / Top-1 / Top-K avg)
                                          │      C. EXCEL + CHART của seed
                                          │      D. XÓA checkpoint + dọn thư mục
                                          └── E. GỘP 5 seed → mean ± std + chart + README
```

## A. Train một seed

Model: `Config.MODEL` (yolov8s-seg.pt). Loss: **mặc định Ultralytics**
(`v8SegmentationLoss` = box + mask + cls BCE + DFL). Optimizer: SGD, `lr0=5e-3`,
`lrf=0.01`, warmup 5 epoch, cosine LR; `momentum=0.937` và `weight_decay=5e-4`
**giữ nguyên mặc định Ultralytics**. AMP bật (H100 → bfloat16).

Ba callback được gắn thêm:

| Callback | Thời điểm | Việc làm |
|---|---|---|
| `cap_val_dataloader_workers` | `on_pretrain_routine_start` | ép val dataloader dùng đúng `WORKERS` (Ultralytics vốn dùng `workers × 2`) |
| `TopKCheckpointManager.on_train_start` | trước epoch 1 | `ema.enabled = False` → **tắt EMA** |
| `.on_train_epoch_end` | sau optimizer step cuối của epoch | copy 1:1 raw `state_dict` sang module validation riêng |
| `.on_model_save` | sau khi Ultralytics lưu `epochN.pt` | ghi đè bằng **raw FP32** model + ghi fitness + **prune** ngoài Top-5 |

Vòng lặp mỗi epoch (do Ultralytics chạy):

```text
train epoch  →  sync raw weights sang model validation  →  validate trên split `val`
             →  fitness = 0.1·mAP50(B) + 0.9·mAP50-95(B) + 0.1·mAP50(M) + 0.9·mAP50-95(M)
             →  best.pt nếu fitness cao nhất  →  early stopping (patience=10)
             →  lưu epochN.pt (raw FP32) → giữ đúng 5 file fitness cao nhất
```

Kết thúc: trên disk còn `best.pt`, `last.pt`, 5 file `epochN.pt` và
`cwa_checkpoints.json` (bảng xếp hạng).

## B. Đánh giá trên `test` (`run_method_evaluation`)

`rank_checkpoints()` đọc JSON → list `(file, fitness, epoch)` giảm dần.

| # | Strategy | Weights | BN |
|---|---|---|---|
| 1 | `Baseline (best.pt)` | best.pt (FP16) | nguyên |
| 2 | `Top-1 (best raw ckpt)` | `ranked[0]` raw FP32, **không average** | nguyên |
| 3 | `CWA (Top-2 avg)` | average 2 ckpt đầu | **recalibrate** |
| 4 | `CWA (Top-3 avg)` | average 3 ckpt đầu | **recalibrate** |
| 5 | `CWA (Top-4 avg)` | average 4 ckpt đầu | **recalibrate** |
| 6 | `CWA (Top-5 avg)` | average 5 ckpt đầu | **recalibrate** |

Mỗi dòng 3–6 chạy đúng 3 bước:

```text
1) average_checkpoints(ranked[:K])
      w_avg = (1/K) · Σ wₜ            ← uniform element-wise, KHÔNG EMA/decay
      áp dụng cho: conv/linear weight, bias, γ và β của BN
      GIỮ NGUYÊN từ ckpt hạng 1: running_mean, running_var, num_batches_tracked

2) update_bn_stats(avg.pt, data)
      reset_running_stats() cho mọi BN, momentum = None (cumulative average)
      model.eval(), riêng BN .train()
      lặp 100 batch của split TRAIN trong torch.no_grad()   ← chỉ forward,
      không backward, không optimizer step
      augmentation khớp giai đoạn train của checkpoint (BN_UPDATE_CLOSE_MOSAIC="auto")

3) model.val(split="test")  →  SegmentMetrics (mask P, R, mAP50, mAP75, mAP50-95)
      → xóa avg.pt ngay
```

## C–D. Kết quả của seed rồi dọn

`seed_<N>_results.xlsx` (Summary | PerEpoch | Checkpoints), chart
`checkpoint_selection.png`, sau đó xóa toàn bộ `weights/`, xóa ảnh mẫu
(`train_batch*`, `val_batch*`, `labels*`), gom chart vào `charts/` và
`results.csv` + `args.yaml` vào `logs/`.

Một seed lỗi không dừng cả experiment — ghi nhận lỗi rồi chạy tiếp seed sau.

## E. Gộp 5 seed

`summary/yolov8s_exp1_summary.xlsx`:

- `MeanStd` — **mean ± std (ddof=1) của 5 seed cho từng strategy** ← bảng chính
- `DeltaVsBaseline` — Δ ghép cặp theo seed vs Baseline + số seed thắng
- `PerSeed`, `PerClass_MeanStd`, `PerClass_PerSeed`, `Checkpoints`, `RunInfo`

`summary/charts/`: bar mean±std, **mAP theo K**, per-seed lines, Δ paired,
Δ theo class. `README.md` ở gốc experiment có bảng mean ± std dạng markdown.

## Thư mục cuối cùng

```text
results/segmentation/yolov8s_exp1/
├── README.md
├── experiment_config.json
├── summary/{yolov8s_exp1_summary.xlsx, charts/}
└── seeds/seed_<N>/{seed_<N>_results.xlsx, charts/, logs/}
```

Không có file `.pt` nào được giữ lại.

## Các chốt kiểm soát dữ liệu

- Checkpoint được **chọn** bằng fitness trên split `val` (401 ảnh).
- BN được **recalibrate** bằng split `train` (3156 ảnh).
- Số liệu **báo cáo** lấy trên split `test` (276 ảnh), không tham gia bước nào ở trên.
