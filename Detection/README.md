# CWA — Object Detection (Ultralytics YOLO + Pascal VOC)

Pipeline huấn luyện object detection bằng Ultralytics, chọn Top-K checkpoint
trên validation rồi uniform average raw learnable parameters. Phiên bản thư
viện được khóa tại `ultralytics==8.3.152`.

## Data split và chống test leakage

`VOC.yaml` gốc của Ultralytics khai báo:

| Split | Ảnh | Nguồn |
|---|---|---|
| train | 16,551 | VOC2007 trainval (5,011) + VOC2012 trainval (11,540) |
| val | 4,952 | VOC2007 test |
| test | 4,952 | **cùng ảnh VOC2007 test** — không có holdout riêng |

Nghĩa là `val ≡ test`. Method dựa trên Top-K checkpoint chọn theo validation
nên nếu dùng nguyên bộ này thì checkpoint được chọn trên chính tập báo cáo →
leakage. `dataset.py` vì thế tách validation độc lập ra khỏi train với
`VAL_RATIO=0.1`:

| Split | Ảnh | Vai trò |
|---|---|---|
| `train'` | 14,896 | Train model và re-estimate BatchNorm sau averaging |
| `val'` | 1,655 | Tính fitness, early stopping, chọn best checkpoint, rank Top-K |
| `test` | 4,952 | VOC2007 test, giữ nguyên; chỉ dùng báo cáo cuối |

Việc tách dùng chính seed của run (`random.Random(seed).shuffle`), nên mỗi seed
có một split `train'/val'` khác nhau — biến thiên của data split được tính vào
std giữa các seed. File split được cache theo `(seed, ratio)` tại dataset root
(`holdout_seed<S>_val10.{yaml,txt}`) nên chạy lại cùng seed sẽ tái lập y hệt.

Không được nhìn kết quả test để chọn K. Code chỉ in tất cả K trong
`TOP_K_VALUES`; không có dòng nào tự chọn “best strategy” từ test.

## Raw checkpoint và phép averaging

Khi bật CWA:

1. EMA smoothing của Ultralytics được tắt.
2. Sau mỗi epoch, raw training weights được copy chính xác sang một module
   validation riêng. Validation, fitness, early stopping và `best.pt` đều theo
   raw weights.
3. Callback lưu raw model ở FP32 vào `epochN.pt`, gắn
   `weight_source="raw"` và chỉ giữ `KEEP_TOP_K_CHECKPOINTS` checkpoint có
   validation fitness cao nhất.
4. Với từng K đã định trước, code thực hiện:

   \[
   \mathbf{w}_{avg}=\frac{1}{K}\sum_{t\in\mathcal{S}_K}\mathbf{w}_t
   \]

   Đây là uniform element-wise average của tất cả learnable parameters:
   convolution/linear weights, bias và affine parameters (γ, β) của BatchNorm.
   **Không phải EMA**: không có hệ số decay, không phụ thuộc thứ tự, mỗi
   checkpoint đóng góp đúng `1/K` bất kể rank hay epoch.
5. Không average state không learnable như `running_mean`, `running_var`,
   `num_batches_tracked`, anchors, stride hoặc cache. Những state này không nằm
   trong \(\mathbf{w}\); chúng được giữ từ checkpoint tốt nhất rồi ước lượng
   lại (xem mục BatchNorm bên dưới).
6. Average và checkpoint kết quả giữ FP32; phép cộng thực hiện ở float64 rồi
   mới hạ về dtype gốc, nên kết quả không phụ thuộc thứ tự cộng và không tích
   lũy sai số làm tròn theo K.
7. Trước khi average, code kiểm tra mọi checkpoint có cùng tập key và cùng
   shape; lệch là dừng chứ không im lặng bỏ qua.

## BatchNorm recalibration

Sau khi average, weights của các layer trước BN đã đổi nhưng `running_mean` /
`running_var` vẫn là của checkpoint cũ → phân phối activation lệch thống kê.
`evaluate.update_bn_stats()` ước lượng lại (tương đương
`torch.optim.swa_utils.update_bn`):

1. `reset_running_stats()` và đặt `momentum = None` → BN dùng **cumulative
   moving average**, tức trung bình chính xác trên toàn bộ batch đã đi qua,
   không phụ thuộc thứ tự batch, không có quán tính.
2. **Lặp qua training data ở chế độ forward-only**:
   - toàn bộ nằm trong `torch.no_grad()` → không dựng graph, không backward;
   - không tạo optimizer, không có `optimizer.step()`;
   - `model.requires_grad_(False)` cho toàn mạng;
   - `model.eval()` cho toàn mạng, riêng các module BN gọi `.train()` để chúng
     tích lũy mean/var;
   - kết thúc vòng lặp code **assert** mọi `parameter.grad is None`, sai là
     raise — đây là bằng chứng chạy được rằng không có cập nhật gradient nào.
3. Chỉ `running_mean` / `running_var` / `num_batches_tracked` thay đổi;
   learnable parameters giữ nguyên đúng giá trị vừa average.

Dữ liệu dùng là **split `train'`**, không bao giờ chạm `val'` hay `test`.
Dataloader dùng `mode='train'` (`BN_UPDATE_AUGMENT=True`) để phân phối ảnh khớp
đúng phân phối mà BN stats gốc được tích lũy trên đó lúc train.

Vì CWA được BN-recalibrate còn baseline thì không, `BN_UPDATE_CONTROL=True`
thêm một dòng **“Baseline + BN recal”** — baseline đã BN-recalibrate — để tách
phần cải thiện do *averaging* khỏi phần do *hiệu chỉnh BN*. Nếu không có control
này thì không thể quy kết cải thiện cho method.

Detection fitness trong Ultralytics 8.3.152 là:

\[
\text{fitness}=0.1\,\mathrm{mAP}_{50}+0.9\,\mathrm{mAP}_{50:95}
\]

`trainer.fitness` chính là criterion được dùng để rank checkpoint và điều
khiển early stopping.

## Classification loss (`LOSS_FUNCTION`)

`'bce'` giữ nguyên mặc định Ultralytics; `'focal'` thay `criterion.bce` bằng
`FocalBCE` (BCE × `(1-p_t)^γ` × α-factor, giữ shape element-wise).

Lưu ý về cách cài: **không** patch được `yolo_model.model` trước khi train.
`Model.train()` dựng một `DetectionModel` HOÀN TOÀN MỚI:

```python
self.trainer.model = self.trainer.get_model(weights=..., cfg=self.model.yaml)
```

nên mọi thứ gắn lên module cũ đều bị vứt đi và training âm thầm chạy bằng BCE.
Vì vậy loss được gán ở callback `on_train_start` (lúc model đã ở đúng device —
`v8DetectionLoss` chụp device khi khởi tạo), cho **cả** `trainer.model` lẫn
`trainer.ema.ema` để cột `val/cls_loss` trong `results.csv` cùng thang đo với
`train/cls_loss`. Loss được gán vào `model.criterion` chứ không phải
monkeypatch `init_criterion`, vì gắn bound method lên instance làm hỏng pickle
của checkpoint.

`assert_cls_loss_installed` chạy ngay sau đó và **raise** nếu loss thực tế
không phải `FocalBCE` — một bản Ultralytics tương lai làm hỏng cơ chế này sẽ bị
phát hiện ngay thay vì hỏng âm thầm cả experiment.

## Checkpoint lifecycle

Checkpoint chỉ là artifact tạm:

- Trong train: giữ tối đa Top-K raw epoch checkpoint, cộng `best.pt`/`last.pt`
  do trainer cần.
- Mỗi averaged checkpoint bị xóa ngay sau khi `model.val()` hoàn tất.
- Sau khi đã export metrics/Excel (và model deploy nếu bật), toàn bộ thư mục
  `weights/` của seed bị xóa, kể cả khi evaluation gặp lỗi.
- Sau đó `tidy_run_dir()` rút gọn thư mục seed xuống còn **Excel + `charts/`**:
  ảnh/plot dồn vào `charts/training/`, `results.csv` và `args.yaml` bị xóa
  (nội dung đã nằm nguyên trong sheet `PerEpoch` và `RunInfo`), ảnh debug
  `train_batch*`/`val_batch*` bị xóa. File lạ không nhận diện được thì **giữ
  nguyên**, không xóa mù.

Chạy `--keep-checkpoints` để tắt cả hai bước trên khi cần debug. Lệnh
`strategies`/`eval` từ run cũ chỉ dùng được nếu run đó đã train với
`--keep-checkpoints` (hoặc `DELETE_CHECKPOINTS_AFTER_RUN=False`).

## Cài đặt và chạy

```bash
pip install -r requirements.txt
python main.py train --exp-name yolov8s_exp1
```

`--exp-name` và `--exp_name` đều hợp lệ. Tên được dùng nguyên văn cho thư mục
kết quả; một lệnh chạy trọn 5 seed trong `RANDOM_SEED` và sinh:

```text
results/detection/yolov8s_exp1/
├── SUMMARY.xlsx              ★ mean ± std của cả 5 seed × mọi strategy
├── experiment_config.json    snapshot config lúc chạy
├── charts/
│   ├── 01_topk_curve_mAP50-95.png    mAP theo K + dải ±std + mức baseline
│   ├── 02_method_mAP50-95.png      dot plot mean ± std mọi strategy
│   ├── 03_method_mAP50.png
│   ├── 04_delta_vs_baseline.png      Δ vs baseline, kèm chấm từng seed
│   └── 05_per_seed_paired.png        mỗi seed một đường (paired)
└── seeds/
    ├── seed_1/
    │   ├── results_seed_1.xlsx
    │   └── charts/
    │       ├── training/             curve + confusion matrix trên val'
    │       ├── test_best/            PR/F1 curve + confusion matrix trên test
    │       ├── test_best_bn/
    │       ├── test_top_2/ … test_top_5/
    ├── seed_10/ …
```

Không có file `.pt` nào được giữ lại. Tên experiment đã tồn tại và không rỗng
sẽ bị từ chối để tránh trộn kết quả giữa các lần chạy.

Một seed lỗi (OOM, hỏng data…) **không** làm mất kết quả của các seed đã xong:
code ghi nhận lỗi, chạy tiếp seed sau, và vẫn xuất `SUMMARY.xlsx` từ các seed
thành công cùng danh sách seed thất bại ở cuối log.

Các lệnh khác:

```bash
# Tắt CWA
python main.py train --exp-name baseline_run01 --no-cwa

# Đổi tập K cần báo cáo
python main.py train --exp-name k_sweep01 --top-k 2 3 5 8

# Ablation: tắt BN recalibration
python main.py train --exp-name no_bn01 --no-bn-update

# Giữ checkpoint + run dir nguyên vẹn để debug
python main.py train --exp-name debug01 --keep-checkpoints

# Đánh giá một checkpoint được giữ lại có chủ ý
python main.py eval --weights path/to/model.pt --split test

# Export model deploy
python main.py train --exp-name deploy_run01 --export-after-train
```

## Config quan trọng

| Biến | Ý nghĩa |
|---|---|
| `MODEL` | Detection model, ví dụ `yolov8s.pt` |
| `VAL_RATIO` | Tỉ lệ tách validation độc lập từ train; VOC mặc định là `0.1` |
| `PATIENCE` | Early stopping theo raw validation fitness |
| `RANDOM_SEED` | Int hoặc list seed; list = chạy lần lượt rồi tổng hợp mean ± std |
| `TOP_K_VALUES` | Các K được định trước để báo cáo |
| `KEEP_TOP_K_CHECKPOINTS` | Phải ≥ `max(TOP_K_VALUES)` |
| `BASELINE_FROM_RAW_TOPK` | Baseline lấy từ rank #1 raw FP32 thay vì `best.pt` FP16 |
| `USE_BN_UPDATE` | Re-estimate BN buffers trên `train'` sau averaging |
| `BN_UPDATE_BATCHES` | Số batch forward khi recalibrate BN |
| `BN_UPDATE_AUGMENT` | Dataloader `mode='train'` (khớp phân phối lúc train) |
| `BN_UPDATE_CONTROL` | Thêm dòng ablation “Baseline + BN recal” |
| `DELETE_CHECKPOINTS_AFTER_RUN` | Xóa checkpoint tạm sau mỗi seed |
| `TIDY_RUN_DIR` | Rút gọn thư mục seed xuống còn Excel + charts |
| `MAKE_CHARTS` | Vẽ chart tổng hợp cuối experiment |
| `EVAL_SPLIT` | Split báo cáo cuối, mặc định `test` |
| `EXP_NAME` | Tên experiment ổn định trên server |
| `WORKERS` | Worker của dataloader lúc train |
| `EVAL_WORKERS` | Worker cho `model.val()` + BN recal (mặc định 8) — xem mục RAM |
| `AUTO_LIMIT_WORKERS` | Tự hạ worker theo số CPU SLURM cấp cho job (mặc định `True`) |

Tại sao `BASELINE_FROM_RAW_TOPK=True`: `best.pt` được Ultralytics
`strip_optimizer()` lưu ở FP16, trong khi checkpoint average là FP32. Rank #1
trong bảng ranking là *đúng cùng epoch, cùng weights* với `best.pt` nhưng giữ
FP32, nên dùng nó làm baseline thì hai nhánh đi qua cùng precision và cùng
đường load/eval. Tie-break của ranking (`fitness` giảm dần, hòa thì epoch sớm
hơn) được đặt khớp đúng ngữ nghĩa `best_fitness` của Ultralytics để rank #1
luôn trùng epoch với `best.pt`.

## Output metrics

`seeds/seed_<N>/results_seed_<N>.xlsx` — 5 sheet:

| Sheet | Nội dung |
|---|---|
| `RunInfo` | Toàn bộ hyperparameter + note về data split |
| `Overall` | 1 dòng/strategy: P, R, mAP@0.5, mAP@0.75, mAP@0.5:0.95, fitness, epoch nguồn |
| `PerClass` | AP từng class × strategy |
| `TopK_Checkpoints` | Epoch nào lọt Top-K, fitness `val'`, dùng cho K nào |
| `PerEpoch` | Nguyên `results.csv` của Ultralytics |

`SUMMARY.xlsx` — 7 sheet:

| Sheet | Nội dung |
|---|---|
| `Mean_Std` | ★ 1 dòng/strategy, có sẵn cột `"0.5150 ± 0.0047"` copy thẳng vào paper |
| `PerSeed` | Số liệu thô, 1 dòng/(seed × strategy) |
| `Delta_vs_Baseline` | Δ paired vs baseline: mean ± std, số seed thắng, t-statistic, p-value |
| `PerClass_Mean_Std` | AP từng class, mean ± std trên các seed |
| `PerClass_PerSeed` | AP từng class, số liệu thô |
| `TopK_Checkpoints` | Epoch lọt Top-K ở từng seed |
| `Config` | Snapshot config lúc chạy |

`Delta_vs_Baseline` dùng **paired two-sided t-test**: cùng seed nghĩa là cùng
data split và cùng quá trình train, nên so sánh theo cặp mới đúng thiết kế thí
nghiệm (unpaired sẽ bị nuốt bởi biến thiên giữa các seed, vốn lớn hơn hiệu ứng
nhiều lần). `p` tính bằng regularized incomplete beta, không cần scipy.

## RAM của host (không phải VRAM)

Nhiều seed chạy trong **cùng một process**, nên RAM phải quay về mức cũ sau mỗi
seed. Hai thứ khiến nó không quay về nếu không xử lý — cả hai đã được xử lý
trong `memory.py`:

1. `InfiniteDataLoader` của Ultralytics tạo sẵn `self.iterator` ngay trong
   `__init__` và giữ suốt đời object ⇒ `workers` process con sống tới khi
   loader bị hủy, mỗi worker còn ôm `prefetch_factor`(=2) batch trong hàng đợi.
2. `DetMetrics` mà `model.val()` trả về giữ `on_plot` = bound method của
   validator ⇒ giữ luôn `validator.dataloader` ở (1). Kết quả từng seed được
   giữ tới cuối experiment, nên chỉ cần lưu DetMetrics là mọi dataloader của
   mọi lượt eval đều không bao giờ được thu hồi.

Với `USE_CWA=True`, MỖI seed dựng 3 dataloader lúc train + 6 lượt
`model.val()` + 5 lượt BN recalibration = 14 dataloader. Ở `WORKERS=24` là 336
process con còn sống sau seed đầu tiên, seed thứ hai chết ngay lúc build
dataset (`Killed` / `slurmstepd: ... oom_kill`).

Vì vậy pipeline:

- Trả về `MetricsSnapshot` (chỉ float/str) thay cho `DetMetrics`.
- Đóng dataloader **tường minh** (`_shutdown_workers`) sau mỗi lượt val, mỗi
  lượt BN recal và sau mỗi seed, thay vì phó mặc cho GC.
- Dùng `EVAL_WORKERS` (mặc định 8) cho các lượt sau train: hàng đợi prefetch
  tốn `workers × 2 × batch × 3 × imgsz²` byte — với `WORKERS=24, batch=128,
  imgsz=640` là ~7.5 GB cho **một** dataloader. Đổi giá trị này không ảnh hưởng
  metrics, chỉ ảnh hưởng tốc độ và RAM.

Log đầu mỗi seed in `RSS ... | ... worker process`. Con số này phải **đi ngang**
qua các seed; nếu tăng dần thì còn chỗ giữ dataloader lại.

### Vì sao `WORKERS=16` lại thành 32 worker trong log

Hai hệ số ×2 nằm bên trong Ultralytics 8.3.152:

- `DetectionTrainer.get_dataloader` (`models/yolo/detect/train.py:88`):
  `workers = self.args.workers if mode == "train" else self.args.workers * 2`
- `BaseTrainer._setup_train` (`engine/trainer.py:312`) gọi nó với `batch_size * 2`

Nên lúc train luôn có **hai** loader sống song song, và loader validation tốn
gấp ~4 lần loader train:

| loader | worker | batch | prefetch (imgsz 640) |
|---|---|---|---|
| `train_loader` | `WORKERS` | `BATCH` | 16 × 2 × 128 → 5.0 GB |
| `test_loader` | `WORKERS × 2` | `BATCH × 2` | 32 × 2 × 256 → 20.1 GB |

Cảnh báo `This DataLoader will create 32 worker processes in total` xuất hiện vì
`build_dataloader` chặn worker bằng `os.cpu_count()` — CPU của **cả node** —
trong khi PyTorch so với `os.sched_getaffinity` — CPU **SLURM cấp cho job**.
Node 96 CPU nhưng job xin 16 thì Ultralytics vẫn tạo đủ 32.

`AUTO_LIMIT_WORKERS=True` (mặc định) hạ `WORKERS` xuống `cpu_quota // 2` để
không loader nào vượt số CPU được cấp, và hạ `EVAL_WORKERS` xuống `cpu_quota`.
`validate_config()` in ra con số hiệu lực + ước lượng RAM prefetch.

`CACHE` **không** liên quan tới lỗi này: mặc định đã là `False` (không cache ảnh
vào RAM). Chỉ `CACHE="ram"` mới nạp cả dataset vào RAM — đừng bật trên VOC.

Troubleshooting:

- OOM host (`Killed`, `slurmstepd: oom_kill`, không phải `CUDA out of memory`):
  giảm `--eval-workers` rồi tới `--workers`, hoặc xin thêm `--mem` cho job.
  `validate_config()` in sẵn ước lượng RAM hàng đợi khi con số này vượt 8 GB.
- OOM VRAM (`CUDA out of memory`): giảm `BATCH`/`IMGSZ` hoặc dùng `--batch -1`.
- Không đủ Top-K: số epoch thực tế trước early stopping nhỏ hơn K.
- Run cũ báo không phải raw checkpoint: phải train lại bằng code hiện tại;
  code chủ động không trộn EMA checkpoint với raw averaging.
