# CWA — Instance Segmentation (Ultralytics YOLO-seg + Carparts)

Pipeline này chỉ dùng model YOLO segmentation (`-seg`) qua Ultralytics API.
Mọi cấu hình chính nằm trong `config.py`; `main.py` cung cấp CLI để train,
đánh giá Baseline/2, xuất Excel và export model.

Xem [`FLOW.md`](FLOW.md) để nắm luồng chạy từ đầu đến cuối.

## Những gì giữ nguyên mặc định của Ultralytics

- **Loss**: `v8SegmentationLoss` nguyên bản (box + mask + cls BCE + DFL).
  `LOSS_FUNCTION = "bce"`. Tùy chọn `"focal"` chỉ để ablation.
- **SGD**: `momentum = 0.937`, `weight_decay = 5e-4` (Ultralytics tự scale theo
  batch), `warmup_momentum = 0.8`, `warmup_bias_lr = 0.1`, `nbs = 64`.
  `build_train_args()` không truyền các key này nên trainer dùng default.
- Chỉ override: `optimizer`, `lr0`, `lrf`, `warmup_epochs`, `cos_lr`, `mixup`,
  `copy_paste`.

## Dataloader workers

Ultralytics dùng `workers × 2` cho val dataloader lúc train
(`models/yolo/detect/train.py`), nên đặt `WORKERS = 16` thì val chạy 32 worker
và hay bị `DataLoader worker (pid ...) is killed by signal` hoặc hết `/dev/shm`.
`CAP_VAL_WORKERS = True` (mặc định) buộc val loader dùng đúng `WORKERS`.

## AMP

`AMP = True` áp dụng cho cả training (`amp` của Ultralytics) và forward pass của
BN recalibration. Trên H100 autocast tự chọn `bfloat16`.

## Data split

Mặc định dùng `carparts-seg.yaml` của Ultralytics:

| Split | Số ảnh | Mục đích |
|---|---:|---|
| `train` | 3156 | Huấn luyện và BN recalibration |
| `val` | 401 | Fitness, `best.pt`, early stopping, xếp hạng Top-K |
| `test` | 276 | Báo cáo cuối |

`VAL_RATIO = 0` giữ nguyên ba split độc lập trên. Với dataset custom, phải bảo
đảm `val` không trùng `test`. Nếu cần tự tách validation từ train thì đặt
`VAL_RATIO > 0`; `dataset.py` sẽ tạo split cố định theo seed.

Pipeline dùng `carparts-seg.yaml` được pin ngay trong project thay vì YAML cũ
đóng gói trong `ultralytics==8.3.152`.

## Các strategy được báo cáo

| Strategy | Mô tả |
|---|---|
| `Baseline (best.pt)` | `best.pt` tạm — raw checkpoint có raw-model fitness validation cao nhất (FP16 do `strip_optimizer` của Ultralytics) |
| `Top-1 (best raw ckpt)` | Chính checkpoint đó nhưng **raw FP32, không average, không đụng BN** — baseline K=1 cùng precision với CWA. Bật/tắt bằng `EVAL_TOP1_BASELINE` |
| `CWA (Top-K avg)` | Uniform element-wise average raw weights của Top-K raw checkpoint + BN recalibration |

`Top-1` tồn tại để tách bạch hai nguồn chênh lệch: precision của checkpoint
(FP16 vs FP32) và tác dụng thật của averaging. Nó cũng là điểm K = 1 của đường
cong mAP theo K trong `summary/charts/03_topk_curve_mAP50-95.png`.

Ultralytics segmentation fitness được dùng thống nhất cho `best.pt`, early
stopping và Top-K:

```text
fitness =
    0.1 × mAP50(B) + 0.9 × mAP50-95(B)
  + 0.1 × mAP50(M) + 0.9 × mAP50-95(M)
```

`B` là bounding box, `M` là mask. Precision và Recall không đóng góp vào
fitness.

Với tập checkpoint được chọn là `S_K`, CWA thực hiện:

```text
w_avg = (1 / K) × Σ w_t^raw,  t ∈ S_K
```

Đây là uniform element-wise averaging, không có parameter mới được học trong
quá trình averaging.

## Raw checkpoint averaging

Ultralytics mặc định duy trì EMA, dùng EMA cho validation và chỉ lưu EMA vào
checkpoint. Pipeline này chủ động tắt EMA khi bắt đầu train để phương pháp khớp
raw checkpoint averaging trong paper:

1. Optimizer cập nhật raw `trainer.model`.
2. Cuối mỗi epoch, raw `state_dict` được copy 1:1 sang một module validation
   riêng; validation và fitness dùng đúng snapshot raw đó.
3. Early stopping và `best.pt` dùng raw-model fitness.
4. Mỗi `epochN.pt` lưu raw FP32 model trong field `model`, với `ema=None`.
5. Sau training, chọn Top-K raw checkpoints và uniform-average raw parameters
   ở FP32; averaged checkpoint tạm cũng dùng FP32.
6. Sau khi đã lấy đủ metrics/Excel, tất cả checkpoint tạm được xóa.

Module validation riêng chỉ để cô lập `torch.inference_mode()` khỏi training
model; nó không thực hiện EMA smoothing. Vì vậy không có EMA smoothing trước
phép Top-K averaging.

## Checkpoint và BatchNorm

Khi `USE_CWA=True`:

- `save_period=1` lưu một `epochN.pt` mỗi epoch.
- Callback tắt EMA và lấy raw-model `trainer.fitness`.
- Chỉ giữ `KEEP_TOP_K_CHECKPOINTS` checkpoint có fitness cao nhất.
- Ranking được ghi vào `weights/cwa_checkpoints.json`.
- Raw FP32 **learnable parameters** (conv/linear weight, bias, γ/β của BN) được
  uniform-average; các checkpoint phải cùng kiến trúc, nếu không `average_checkpoints`
  raise ngay.
- BN `running_mean`, `running_var`, `num_batches_tracked` **không** được average
  (đây là population statistics, không phải tham số học được) — chúng giữ nguyên
  từ checkpoint hạng 1 rồi bị `update_bn_stats()` reset và ước lượng lại.
- Mỗi `cwa_topK_avg.pt` được xóa ngay sau `model.val()`.
- Cuối mỗi seed, toàn bộ thư mục `weights/` (`best.pt`, `last.pt`, raw
  `epochN.pt`, ranking JSON còn lại) được xóa trong khối `finally`.

Run cũ dùng EMA hoặc JSON xếp hạng cũ sẽ bị từ chối; cần train lại vì raw
checkpoints tương ứng không tồn tại trong run cũ.

### BN recalibration (`update_bn_stats`)

Sau averaging, weights của các layer trước BN đã đổi nhưng `running_mean` /
`running_var` vẫn là của checkpoint cũ → phân phối activation lệch. Quy trình
khớp đúng `torch.optim.swa_utils.update_bn`:

1. `reset_running_stats()` cho mọi `_BatchNorm`, đặt `momentum = None`
   (cumulative moving average, không phải EMA).
2. Toàn model `eval()`, riêng các BN module `train()` để tích lũy thống kê.
3. Lặp `BN_UPDATE_BATCHES` batch của **split train** trong `torch.no_grad()` —
   **không** backward, **không** optimizer step, chỉ forward.
4. Trả `momentum` về giá trị cũ, ghi đè checkpoint (chỉ BN buffers thay đổi;
   learnable parameters giữ nguyên FP32 vừa average).

`BN_UPDATE_CLOSE_MOSAIC` quyết định augmentation của dataloader dùng để ước
lượng BN. Ultralytics tắt mosaic/mixup/cutmix/copy_paste trong `close_mosaic`
epoch cuối (mặc định 10); nếu ước lượng BN trên ảnh mosaic trong khi checkpoint
được train ở giai đoạn không mosaic thì BN stats sẽ lệch. Mặc định `"auto"` bám
theo epoch của checkpoint hạng 1: nằm trong `close_mosaic` epoch cuối → tắt
mosaic; early stopping sớm hơn → dùng full augmentation.

Dataloader BN được giải phóng tường minh sau khi dùng: `build_dataloader` của
Ultralytics trả `InfiniteDataLoader` giữ worker process sống, nếu không `del`
thì mỗi giá trị K và mỗi seed lại cộng thêm `WORKERS` process.

## Cài đặt

```bash
pip install -r requirements.txt
```

Phiên bản Ultralytics được khóa trong `requirements.txt` vì pipeline sử dụng
callback, checkpoint format và `SegmentMetrics` nội bộ.

## Chạy

Chỉnh `MODEL`, `DATA` và các hyperparameter trong `config.py`, hoặc override qua
CLI:

```bash
# Train tất cả seed trong Config.RANDOM_SEED; tên output cố định, dễ đọc
python main.py train --exp_name carparts_yolov8s_raw_topk_run01

# Đánh giá lại Baseline và CWA của một run
python main.py strategies --run-dir results/segmentation/<experiment>/seeds/seed_42

# Đánh giá một weights segmentation
python main.py eval --weights <run>/weights/best.pt --split test

# Xuất Excel offline từ results.csv
python main.py export --run-dir <run> --output segmentation_results.xlsx

# Export ONNX
python main.py export-model --weights <run>/weights/best.pt --format onnx
```

Pipeline từ chối model detection thông thường như `yolov8s.pt`; hãy dùng model
segmentation như `yolov8s-seg.pt`.

`--exp-name` và `--exp_name` tương đương. Khi truyền tham số này, output là
`results/segmentation/<exp_name>/`; pipeline không ghép timestamp. Một tên đã
tồn tại và không rỗng sẽ bị từ chối để tránh trộn hai thí nghiệm.

## Output

`python main.py train --exp-name yolov8s_exp1` tạo đúng một thư mục mang tên
experiment, bên trong chỉ có Excel, chart và log — **không có checkpoint `.pt`**:

```text
results/segmentation/yolov8s_exp1/
├── README.md                          # bảng mean ± std đọc nhanh, không cần mở Excel
├── experiment_config.json             # snapshot toàn bộ config lúc chạy
├── summary/
│   ├── yolov8s_exp1_summary.xlsx      # kết quả tổng hợp 5 seed
│   └── charts/
│       ├── 01_mAP50-95_by_strategy.png   # bar mean ± std theo strategy
│       ├── 02_mAP50_by_strategy.png
│       ├── 03_topk_curve_mAP50-95.png    # mAP theo K + đường baseline
│       ├── 04_per_seed_mAP50-95.png      # mỗi seed một đường
│       ├── 05_delta_vs_baseline.png      # Δ ghép cặp theo seed + tỉ lệ thắng
│       └── 06_per_class_delta.png        # Δ AP từng class
└── seeds/
    ├── seed_1/
    │   ├── seed_1_results.xlsx        # Summary | PerEpoch | Checkpoints
    │   ├── charts/
    │   │   ├── training_curves.png        # results.png của Ultralytics
    │   │   ├── checkpoint_selection.png   # fitness/epoch + đánh dấu Top-K
    │   │   ├── confusion_matrix*.png
    │   │   └── Box*_curve.png, Mask*_curve.png
    │   └── logs/{results.csv, args.yaml}
    ├── seed_10/ ...
    └── seed_500/
```

### `summary/<exp_name>_summary.xlsx`

| Sheet | Nội dung |
|---|---|
| `MeanStd` | **mean ± std của toàn bộ seed cho từng strategy** — bảng chính để đưa vào paper (`std` là sample std, `ddof=1`) |
| `DeltaVsBaseline` | Δ **ghép cặp theo seed** so với Baseline + số seed mà strategy đó thắng |
| `PerSeed` | mỗi row = 1 seed × 1 strategy |
| `PerClass_MeanStd` | mean ± std AP theo class × strategy |
| `PerClass_PerSeed` | AP per class thô |
| `Checkpoints` | epoch nào được chọn vào Top-K ở từng seed và fitness tương ứng |
| `RunInfo` | cấu hình experiment, tiêu chí chọn checkpoint, seed lỗi (nếu có) |

### `seeds/seed_<N>/seed_<N>_results.xlsx`

- `Summary`: run info + overall metrics theo strategy (kèm cột `Δ ... vs S1`) +
  mask AP per class theo strategy.
- `PerEpoch`: toàn bộ cột train/validation trong `results.csv` (box và mask).
- `Checkpoints`: ranking Top-K theo fitness val'.

Mặc định không còn `weights/` sau khi một seed hoàn tất
(`DELETE_CHECKPOINTS_AFTER_RUN=True`). Checkpoint chỉ tồn tại tạm thời trong lúc
rank, averaging, BN update, evaluation và export tùy chọn. Muốn giữ hoặc deploy
model `.pt` thì phải đổi policy này trước khi train.

Ảnh mẫu Ultralytics dump ra (`train_batch*.jpg`, `val_batch*.jpg`, `labels*.jpg`)
bị xóa mặc định (`KEEP_SAMPLE_IMAGES=False`) vì chúng nặng và không phải chart.
Plot của từng lần `model.val()` theo strategy tắt mặc định
(`SAVE_EVAL_PLOTS=False`); khi bật, chúng nằm trong `seeds/seed_<N>/eval_plots/`
chứ không rơi vào `runs/segment/valN` như mặc định của Ultralytics.

Một seed lỗi không làm hỏng cả experiment: pipeline ghi lại lỗi, chạy tiếp các
seed còn lại và liệt kê seed thất bại trong `README.md` + sheet `RunInfo`.

## Config quan trọng

| Nhóm | Biến |
|---|---|
| Model/data | `MODEL`, `DATA`, `VAL_RATIO` |
| Training | `EPOCHS`, `IMGSZ`, `BATCH`, `DEVICE`, `PATIENCE`, `RANDOM_SEED` |
| Optimizer/LR | `OPTIMIZER`, `LR0`, `LRF`, `WARMUP_EPOCHS`, `COS_LR` |
| Augmentation | `MIXUP`, `COPY_PASTE`, `EXTRA_TRAIN_ARGS` |
| CWA | `TOP_K_VALUES`, `KEEP_TOP_K_CHECKPOINTS`, `EVAL_TOP1_BASELINE` |
| BN recalibration | `USE_BN_UPDATE`, `BN_UPDATE_BATCHES`, `BN_UPDATE_CLOSE_MOSAIC`, `AMP` |
| Evaluation | `EVAL_SPLIT`, `CONF`, `IOU` |
| Output | `PROJECT`, `EXP_NAME`, `DELETE_CHECKPOINTS_AFTER_RUN`, `KEEP_SAMPLE_IMAGES`, `SAVE_EVAL_PLOTS`, `MAKE_CHARTS` |

`PATIENCE` đếm số epoch không có fitness cải thiện nghiêm ngặt; early stopping
không dựa trên validation loss.

`RANDOM_SEED` nhận list (mặc định `[1, 10, 42, 100, 500]`); mỗi seed là một run
độc lập trong `seeds/seed_<N>/` và tất cả được gộp vào `summary/`.
