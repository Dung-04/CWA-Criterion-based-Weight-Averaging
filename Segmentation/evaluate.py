"""
Evaluation cho CWA - Instance Segmentation (Ultralytics YOLO + Carparts).

- Metrics đọc TRỰC TIẾP từ SegmentMetrics của Ultralytics (results.seg.*:
  map50, map, mp, mr, per-class AP, fitness) — không tự tính lại mask AP.
- Baseline: best.pt raw model có fitness cao nhất trên val'.
- Top-1 baseline: chính raw FP32 checkpoint hạng 1 (không average) — điểm K=1.
- CWA: uniform element-wise average raw weights của Top-K raw checkpoint.
- Excel per-seed (Summary / PerEpoch / Checkpoints) + Excel tổng hợp nhiều seed
  (MeanStd / PerSeed / DeltaVsBaseline / per-class) + chart PNG.
"""
import json
import math
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from config import Config

# File ranking do TopKCheckpointManager (train.py) ghi trong <run_dir>/weights/
RANKING_FILE = "cwa_checkpoints.json"

# Tên strategy — charts.strategy_k() parse "Top-<K>" từ các tên này.
BASELINE_NAME = "Baseline (best.pt)"
TOP1_NAME = "Top-1 (best raw ckpt)"


def cwa_name(k):
    return f"CWA (Top-{int(k)} avg)"


INDEPENDENT_VAL_NOTE = (
    "Validation set (val') ĐỘC LẬP được tách từ train theo VAL_RATIO — chỉ dùng để "
    "chọn best.pt / rank Top-K checkpoint. Test set (hold-out), giữ nguyên "
    "làm hold-out, chỉ dùng cho báo cáo cuối."
)
ORIGINAL_SPLIT_NOTE = (
    "VAL_RATIO=0: dùng nguyên các split trong data.yaml. Checkpoint được chọn "
    "trên val và chỉ báo cáo cuối trên test; cần bảo đảm hai split độc lập. "
    "carparts-seg.yaml mặc định đáp ứng điều kiện này."
)

METRIC_COLUMNS = ["mAP@0.5", "mAP@0.5:0.95", "mAP@0.75", "Precision", "Recall", "Fitness"]
HEADLINE_METRIC = "mAP@0.5:0.95"


def _slug(text):
    """Tên strategy → slug an toàn cho thư mục/file."""
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "run"


# ==================== Metrics extraction (từ SegmentMetrics) ====================

def extract_overall_metrics(metrics):
    """
    Trả về dict metrics overall; ưu tiên mask metrics của SegmentMetrics.
    - Segmentation: extract từ metrics.seg.*
    - Detection: extract từ metrics.box.*
    """
    if hasattr(metrics, "seg") and metrics.seg is not None:
        source = metrics.seg
    elif hasattr(metrics, "box") and metrics.box is not None:
        source = metrics.box
    else:
        raise ValueError(
            "Metrics object không có nhánh 'seg' lẫn 'box' — không thể trích xuất "
            f"metrics từ {type(metrics).__name__}."
        )

    overall = {
        "Precision": float(source.mp),
        "Recall": float(source.mr),
        "mAP@0.5": float(source.map50),
        "mAP@0.75": float(source.map75),
        "mAP@0.5:0.95": float(source.map),
    }

    # SegmentMetrics.fitness = box.fitness() + seg.fitness(), mỗi nhánh dùng
    # 0.1*mAP50 + 0.9*mAP50-95 (Ultralytics).
    fitness = getattr(metrics, "fitness", None)
    if fitness is not None:
        overall["Fitness"] = float(fitness)
    return overall


def extract_per_class_metrics(metrics):
    """List dict per-class (P, R, AP@0.5, AP@0.5:0.95), ưu tiên mask metrics."""
    if hasattr(metrics, "seg") and metrics.seg is not None:
        source = metrics.seg
    elif hasattr(metrics, "box") and metrics.box is not None:
        source = metrics.box
    else:
        return []

    names = getattr(metrics, "names", {}) or {}
    rows = []
    for i, class_idx in enumerate(getattr(source, "ap_class_index", [])):
        class_idx = int(class_idx)
        p, r, ap50, ap = source.class_result(i)
        rows.append({
            "Class ID": class_idx,
            "Class": str(names.get(class_idx, class_idx)),
            "Precision": float(p),
            "Recall": float(r),
            "AP@0.5": float(ap50),
            "AP@0.5:0.95": float(ap),
        })
    return rows


def print_detection_metrics(metrics, header="SEGMENTATION EVALUATION RESULTS"):
    """In metrics segmentation ra console (format banner giống repo gốc)."""
    overall = extract_overall_metrics(metrics)
    per_class = extract_per_class_metrics(metrics)

    print("\n" + "=" * 70)
    print(f" {header}")
    print("=" * 70)
    for key, value in overall.items():
        print(f"  {key:<14}: {value:.4f}")

    if per_class:
        print("-" * 70)
        print(f"  {'Class':<18} {'Precision':>10} {'Recall':>10} {'AP@0.5':>10} {'AP@0.5:0.95':>12}")
        for row in per_class:
            print(
                f"  {row['Class']:<18} {row['Precision']:>10.4f} {row['Recall']:>10.4f} "
                f"{row['AP@0.5']:>10.4f} {row['AP@0.5:0.95']:>12.4f}"
            )
    print("=" * 70)
    return overall, per_class


# ==================== model.val() wrapper ====================

def build_val_args(data, split=None, project=None, name=None, plots=None):
    """
    Map Config → kwargs của model.val().

    project/name được truyền vào để artefact của model.val() (confusion matrix,
    PR curve...) nằm trong thư mục experiment thay vì rơi vào ./runs/segment/valN
    mặc định của Ultralytics.
    """
    val_args = {
        "data": str(data),
        "imgsz": int(Config.IMGSZ),
        "workers": int(Config.WORKERS),
    }
    # auto-batch (-1) chỉ dành cho train → khi val dùng mặc định nếu BATCH=-1
    if int(Config.BATCH) > 0:
        val_args["batch"] = int(Config.BATCH)
    if Config.DEVICE is not None:
        val_args["device"] = Config.DEVICE
    if split:  # None = dùng split mặc định của data.yaml ('val')
        val_args["split"] = split
    if Config.CONF is not None:
        val_args["conf"] = float(Config.CONF)
    if Config.IOU is not None:
        val_args["iou"] = float(Config.IOU)
    if project is not None:
        val_args["project"] = str(project)
        val_args["name"] = str(name or "val")
        val_args["exist_ok"] = True
    if plots is not None:
        val_args["plots"] = bool(plots)
    return val_args


def evaluate_weights(weights, data, split=None, header=None, project=None, name=None, plots=None):
    """Chạy model.val() với weights segmentation đã train và trả về SegmentMetrics."""
    from ultralytics import YOLO

    if not weights:
        raise ValueError(
            "Cần weights đã train để eval: truyền --weights path/to/best.pt "
            "(hoặc set Config.MODEL trỏ tới weights đã train)."
        )

    model = YOLO(str(weights))
    if getattr(model, "task", None) != "segment":
        raise ValueError(
            f"Pipeline này chỉ hỗ trợ instance segmentation, nhưng {weights!s} "
            f"có task={getattr(model, 'task', None)!r}."
        )
    metrics = model.val(**build_val_args(data, split, project=project, name=name, plots=plots))
    print_detection_metrics(
        metrics, header=header or f"EVALUATION (split={split or 'default'}) — {Path(str(weights)).name}"
    )
    return metrics


# ==================== CWA: ranking + weight averaging ====================

def rank_checkpoints(run_dir):
    """
    Rank các checkpoint epoch còn trên disk theo fitness trên val' (giảm dần).
    Checkpoint có fitness cao nhất = tốt nhất, đúng tiêu chí best.pt và early
    stopping của Ultralytics.

    Nguồn chính: cwa_checkpoints.json (TopKCheckpointManager ghi lúc train).
    Fallback: tính đúng SegmentMetrics fitness từ các cột B/M trong results.csv
    khi run dir được copy từ máy khác mà thiếu file JSON.

    Returns:
        list[(Path, fitness, epoch)] sorted theo fitness giảm dần.
    """
    weights_dir = Path(run_dir) / "weights"
    ranking_path = weights_dir / RANKING_FILE
    records = []

    if ranking_path.exists():
        data = json.loads(ranking_path.read_text(encoding="utf-8"))
        for fname, info in data.items():
            path = weights_dir / fname
            if path.exists():
                if "fitness" not in info or info.get("weight_source") != "raw":
                    raise ValueError(
                        f"{ranking_path} không chứa raw-weight ranking hiện tại. "
                        "Cần train lại run này với code raw averaging."
                    )
                records.append((path, float(info["fitness"]), int(info["epoch"])))
    else:
        try:
            df = read_results_csv(run_dir)
        except FileNotFoundError:
            return []
        fitness_columns = (
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "metrics/mAP50(M)",
            "metrics/mAP50-95(M)",
        )
        missing = [column for column in fitness_columns if column not in df.columns]
        if missing:
            raise ValueError(
                "Không thể khôi phục segmentation fitness từ results.csv; "
                f"thiếu cột: {missing}"
            )

        fitness_by_epoch = {}
        for _, row in df.iterrows():
            try:
                fitness = (
                    0.1 * float(row["metrics/mAP50(B)"])
                    + 0.9 * float(row["metrics/mAP50-95(B)"])
                    + 0.1 * float(row["metrics/mAP50(M)"])
                    + 0.9 * float(row["metrics/mAP50-95(M)"])
                )
                if math.isfinite(fitness):
                    fitness_by_epoch[int(row["epoch"])] = fitness
            except (TypeError, ValueError, KeyError):
                continue

        for path in weights_dir.glob("epoch*.pt"):
            digits = re.sub(r"\D", "", path.stem)
            if not digits:
                continue
            file_epoch = int(digits)
            # Tên file epoch{N}.pt đánh số 0-based, cột epoch results.csv 1-based
            fitness = fitness_by_epoch.get(file_epoch + 1, fitness_by_epoch.get(file_epoch))
            if fitness is not None:
                records.append((path, float(fitness), file_epoch))

    # Fitness cao nhất đứng đầu; nếu hòa thì ưu tiên epoch mới hơn.
    records.sort(key=lambda r: (r[1], r[2]), reverse=True)
    return records


def ranking_table(ranked):
    """list[(path, fitness, epoch)] → list dict để ghi ra Excel."""
    return [
        {
            "Rank": i + 1,
            "Checkpoint": path.name,
            # epoch trong file name là 0-based; cột 'epoch' của results.csv là 1-based
            "Epoch (0-based)": epoch,
            "Epoch (results.csv)": epoch + 1,
            "Fitness (val)": round(float(fitness), 6),
        }
        for i, (path, fitness, epoch) in enumerate(ranked)
    ]


def average_checkpoints(ckpt_paths, output_path):
    """
    Uniform element-wise average RAW model parameters của nhiều checkpoint YOLO.

        w_avg = (1/K) * Σ_{t ∈ S_K} w_t          (element-wise, uniform)

    KHÔNG phải EMA: mọi checkpoint có trọng số bằng nhau và không có hệ số
    decay/momentum nào tham gia.

    - Chỉ nhận checkpoint có ``weight_source='raw'`` và raw model trong ``model``.
    - Average đúng các LEARNABLE parameters (float): conv/linear weights, bias,
      γ/β của BatchNorm.
    - KHÔNG average BN running statistics (`running_mean`, `running_var`,
      `num_batches_tracked`) — đây là population stats, không phải learned;
      average chúng làm BN lệch phân phối → giữ nguyên từ checkpoint ĐẦU (đã
      được sort là ckpt có fitness cao nhất) rồi ước lượng lại bằng
      ``update_bn_stats()``.
    - Buffer không phải float (ví dụ num_batches_tracked dạng int) cũng giữ
      nguyên từ checkpoint đầu.

    Ckpt output chỉ chứa averaged raw model (bỏ optimizer, ``ema=None``), load
    lại bằng YOLO(path) như checkpoint thường.
    """
    import torch

    ckpt_paths = [Path(p) for p in ckpt_paths]
    if not ckpt_paths:
        raise ValueError("average_checkpoints() cần ít nhất 1 checkpoint")

    ckpts = [torch.load(str(p), map_location="cpu", weights_only=False) for p in ckpt_paths]
    invalid = [
        str(path)
        for path, ckpt in zip(ckpt_paths, ckpts)
        if ckpt.get("weight_source") != "raw" or ckpt.get("model") is None
    ]
    if invalid:
        raise ValueError(
            "CWA raw averaging chỉ nhận raw checkpoints do code hiện tại tạo. "
            f"Checkpoint không hợp lệ: {invalid}"
        )

    modules = [ckpt["model"].float() for ckpt in ckpts]
    state_dicts = [m.state_dict() for m in modules]
    parameter_keys = set(dict(modules[0].named_parameters()))

    # Mọi checkpoint phải cùng kiến trúc, nếu không phép average là vô nghĩa.
    reference_keys = set(state_dicts[0])
    for path, state_dict in zip(ckpt_paths[1:], state_dicts[1:]):
        if set(state_dict) != reference_keys:
            missing = sorted(reference_keys - set(state_dict))[:5]
            extra = sorted(set(state_dict) - reference_keys)[:5]
            raise ValueError(
                f"Checkpoint {path.name} có state_dict khác kiến trúc "
                f"(thiếu: {missing}, thừa: {extra})."
            )
        for key, tensor in state_dicts[0].items():
            if state_dict[key].shape != tensor.shape:
                raise ValueError(
                    f"Shape không khớp ở '{key}': {tuple(tensor.shape)} vs "
                    f"{tuple(state_dict[key].shape)} ({path.name})."
                )

    def is_bn_stat(key):
        return any(marker in key for marker in ("running_mean", "running_var", "num_batches_tracked"))

    if len(ckpt_paths) == 1:
        # K=1 = chính checkpoint đó, không có phép average nào.
        avg_state = {k: v.clone() for k, v in state_dicts[0].items()}
        n_averaged = 0
    else:
        avg_state = {}
        n_averaged = 0
        for key, ref_tensor in state_dicts[0].items():
            if key not in parameter_keys or is_bn_stat(key) or not ref_tensor.dtype.is_floating_point:
                # Giữ nguyên từ checkpoint đầu (đã sort theo fitness giảm dần)
                avg_state[key] = ref_tensor.clone()
            else:
                stacked = torch.stack([sd[key].float() for sd in state_dicts])
                avg_state[key] = stacked.mean(dim=0).to(ref_tensor.dtype)
                n_averaged += 1

    merged_module = modules[0]
    merged_module.load_state_dict(avg_state, strict=True)
    merged_module = merged_module.float().eval()
    for parameter in merged_module.parameters():
        parameter.requires_grad_(False)
    if hasattr(merged_module, "criterion"):
        merged_module.criterion = None

    torch.save(
        {
            # Giữ FP32 để không quantize kết quả raw averaging sau khi đã tính
            # trung bình ở FP32. YOLO(path) vẫn load checkpoint bình thường.
            "model": merged_module,
            "ema": None,
            "optimizer": None,
            "epoch": -1,
            "best_fitness": None,
            "train_args": ckpts[0].get("train_args", {}),
            "source_checkpoints": [p.name for p in ckpt_paths],
            "date": datetime.now().isoformat(timespec="seconds"),
            "weight_source": "raw_topk_average",
        },
        str(output_path),
    )
    print(
        f"      ✓ Uniform average {len(ckpt_paths)} raw ckpt "
        f"({n_averaged} parameter tensors, BN running stats giữ nguyên): "
        f"{[p.name for p in ckpt_paths]}"
    )
    return Path(output_path)


def close_mosaic_epochs():
    """Số epoch cuối mà Ultralytics tắt mosaic (mặc định 10)."""
    from ultralytics.utils import DEFAULT_CFG_DICT

    value = (Config.EXTRA_TRAIN_ARGS or {}).get(
        "close_mosaic", DEFAULT_CFG_DICT.get("close_mosaic", 10)
    )
    return int(value or 0)


def resolve_mosaic_closed(ranked=None):
    """
    Quyết định BN update nên dùng ảnh mosaic hay không.

    ``Config.BN_UPDATE_CLOSE_MOSAIC``:
      - True   : luôn tắt mosaic khi update BN.
      - False  : luôn dùng full train augmentation.
      - "auto" : bám theo giai đoạn train mà checkpoint hạng 1 thuộc về. Nếu
                 checkpoint tốt nhất rơi vào ``close_mosaic`` epoch cuối (lúc
                 Ultralytics đã tắt mosaic/mixup/cutmix/copy_paste) thì BN cũng
                 phải được ước lượng trên phân phối ảnh đó; ngược lại (early
                 stopping sớm) thì dùng full augmentation.
    """
    setting = Config.BN_UPDATE_CLOSE_MOSAIC
    if isinstance(setting, bool):
        return setting
    if str(setting).lower() != "auto":
        raise ValueError("BN_UPDATE_CLOSE_MOSAIC phải là True, False hoặc 'auto'")

    close_mosaic = close_mosaic_epochs()
    if close_mosaic <= 0 or not ranked:
        return close_mosaic > 0
    # epoch trong ranked là 0-based; giai đoạn mosaic-closed bắt đầu ở
    # epoch index (EPOCHS - close_mosaic).
    return int(ranked[0][2]) >= int(Config.EPOCHS) - close_mosaic


def _build_bn_dataloader(data_yaml, stride, mosaic_closed):
    """
    Dataloader train dùng để ước lượng lại BN statistics.

    Ultralytics tắt mosaic/mixup/cutmix/copy_paste trong ``close_mosaic`` epoch
    cuối (mặc định 10), mà Top-K checkpoint thường nằm trong giai đoạn này. Nếu
    ước lượng BN trên ảnh mosaic trong khi checkpoint được train không có mosaic
    thì phân phối activation lệch → BN stats sai. ``mosaic_closed`` do
    ``resolve_mosaic_closed()`` quyết định.
    """
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import DEFAULT_CFG_DICT

    data = check_det_dataset(str(data_yaml))
    batch = int(Config.BATCH) if int(Config.BATCH) > 0 else 16

    overrides = {
        "imgsz": int(Config.IMGSZ),
        "mode": "train",
        "mixup": float(Config.MIXUP),
        "copy_paste": float(Config.COPY_PASTE),
    }
    # EXTRA_TRAIN_ARGS có thể chứa key không thuộc DEFAULT_CFG (get_cfg sẽ raise)
    # hoặc key không liên quan tới dataloader → chỉ lấy phần hợp lệ.
    for key, value in (Config.EXTRA_TRAIN_ARGS or {}).items():
        if key in DEFAULT_CFG_DICT:
            overrides[key] = value
    overrides["task"] = "segment"

    if mosaic_closed:
        overrides.update({"mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0})
        regime = f"mosaic-closed (khớp {close_mosaic_epochs()} epoch cuối của train)"
    else:
        regime = "full train augmentation"

    cfg = get_cfg(overrides=overrides)
    dataset = build_yolo_dataset(
        cfg, data["train"], batch, data, mode="train", rect=False, stride=stride
    )
    loader = build_dataloader(dataset, batch, workers=int(Config.WORKERS), shuffle=True, rank=-1)
    return loader, dataset, batch, regime


def update_bn_stats(weights_path, data_yaml, num_batches=None, device=None, mosaic_closed=None):
    """
    Re-estimate BN running statistics của model đã average — tương đương
    ``torch.optim.swa_utils.update_bn`` / ``update_bn()`` của nhánh
    CWA_TinyImageNet, thích ứng cho YOLO.

    Vì sao: sau average_checkpoints() BN running_mean/running_var được giữ từ
    ckpt tốt nhất (không average) nhưng weights của các layer trước BN đã đổi
    → phân phối activation lệch với running stats cũ. Chạy forward pass trên
    train (BN ở mode 'train', momentum=None → cumulative moving average) để
    tính lại running stats khớp weights mới. Ghi đè lại `weights_path`.

    KHÔNG có gradient nào được tính (``torch.no_grad()``) và không có optimizer
    step nào: đây thuần túy là forward pass để tích lũy thống kê.

    Args:
        weights_path: file .pt đã average (do `average_checkpoints` sinh ra).
        data_yaml: path data yaml (holdout hoặc gốc) — dùng SPLIT TRAIN để
                   ước lượng BN, không đụng val'/test.
        num_batches: số batch forward (mặc định Config.BN_UPDATE_BATCHES).
        device: None = auto GPU nếu có.
        mosaic_closed: None = tự quyết theo Config.BN_UPDATE_CLOSE_MOSAIC.

    Returns:
        weights_path (đã ghi đè với BN stats mới).
    """
    import gc

    import torch
    from torch.nn.modules.batchnorm import _BatchNorm
    from ultralytics import YOLO

    weights_path = Path(weights_path)
    num_batches = int(num_batches or Config.BN_UPDATE_BATCHES)
    if num_batches <= 0:
        raise ValueError("BN_UPDATE_BATCHES phải > 0 khi USE_BN_UPDATE=True")

    yolo = YOLO(str(weights_path))
    model = yolo.model
    bn_modules = [m for m in model.modules() if isinstance(m, _BatchNorm)]
    if not bn_modules:
        print("      (Không tìm thấy BN layer nào, bỏ qua update_bn)")
        return weights_path

    device_obj = torch.device(
        device
        or (f"cuda:{Config.DEVICE}" if isinstance(Config.DEVICE, str) and Config.DEVICE.isdigit()
            else "cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.float().to(device_obj)
    stride = int(max(model.stride)) if hasattr(model, "stride") else 32

    if mosaic_closed is None:
        mosaic_closed = resolve_mosaic_closed()
    loader, dataset, batch_size, regime = _build_bn_dataloader(data_yaml, stride, mosaic_closed)

    # Reset BN running stats + đổi momentum=None (cumulative moving average)
    saved_momentum = {}
    for m in bn_modules:
        saved_momentum[m] = m.momentum
        m.reset_running_stats()
        m.momentum = None

    # Toàn model ở eval() để tắt dropout/random path, riêng BN bật train() để
    # tích lũy running stats — pattern giống update_bn của nhánh classification.
    was_training = model.training
    model.eval()
    for m in bn_modules:
        m.train()

    # Autocast khớp precision train. BN chỉ tích lũy mean/var nên không nhạy
    # cảm với precision, nhưng match train precision cho nhất quán và nhanh hơn.
    if getattr(Config, "AMP", True) and device_obj.type == "cuda":
        autocast_ctx = torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
    else:
        autocast_ctx = torch.autocast(device_type=device_obj.type, enabled=False)

    epochs_equiv = num_batches * batch_size / max(len(dataset), 1)
    print(
        f"      Updating BN stats: {num_batches} batch × {batch_size} ảnh "
        f"(~{epochs_equiv:.1f} epoch train) | aug={regime} | device={device_obj}"
    )
    seen = 0
    try:
        with torch.no_grad(), autocast_ctx:
            # build_dataloader trả InfiniteDataLoader (lặp vô hạn) → BẮT BUỘC break.
            for i, batch_data in enumerate(loader):
                if i >= num_batches:
                    break
                img = batch_data["img"].to(device_obj, non_blocking=True).float() / 255.0
                _ = model(img)
                seen += 1
    finally:
        for m, mom in saved_momentum.items():
            m.momentum = mom
        model.train(was_training)
        # InfiniteDataLoader giữ worker process sống; không giải phóng thì mỗi K
        # và mỗi seed lại cộng thêm WORKERS process → cạn file descriptor/shm.
        del loader, dataset
        gc.collect()

    if seen < num_batches:
        print(f"      ⚠ Chỉ chạy được {seen}/{num_batches} batch cho BN update")

    # Ghi đè weights_path với BN stats mới (giữ nguyên cấu trúc ckpt).
    # Giữ FP32 để không lượng tử hóa các learnable parameters vừa được uniform
    # average ở FP32; bước này chỉ được phép thay đổi BN running statistics.
    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    ckpt["model"] = model.float().cpu().eval()
    ckpt["bn_update"] = {"batches": seen, "batch_size": batch_size, "augmentation": regime}
    torch.save(ckpt, str(weights_path))
    print(f"      ✓ BN stats updated: {weights_path.name}")
    return weights_path


# ==================== Strategy evaluation ====================

def run_method_evaluation(run_dir, data=None, split=None, seed=None, excel_path=None):
    """
    Đánh giá các strategy trên split báo cáo (mặc định Config.EVAL_SPLIT='test'):

      - Baseline (best.pt)      : checkpoint fitness cao nhất, FP16 do
                                    strip_optimizer của Ultralytics.
      - Top-1 (best raw ckpt)     : cùng checkpoint đó nhưng raw FP32, không
                                    average — baseline K=1 cùng precision với
                                    CWA (bật/tắt qua EVAL_TOP1_BASELINE).
      - CWA (Top-K avg)    : uniform average K checkpoint tốt nhất, có
                                    BN recalibration.

    Returns:
        dict {
            "strategies": {tên strategy: SegmentMetrics},
            "checkpoints": list dict ranking Top-K,
            "excel": path Excel đã ghi (hoặc None),
        }
    """
    run_dir = Path(run_dir)
    data = data or resolve_data_from_run(run_dir) or Config.DATA
    split = split or Config.EVAL_SPLIT

    print("\n" + "=" * 70)
    print(f" STRATEGY EVALUATION (split={split or 'default'}, data={data})")
    print("=" * 70)

    results = {}
    eval_root = run_dir / "eval_plots"
    plots = bool(Config.SAVE_EVAL_PLOTS)

    def _val(weights, name, header):
        return evaluate_weights(
            weights, data, split, header=header,
            project=eval_root, name=_slug(name), plots=plots,
        )

    # ----- Baseline: best.pt (fitness cao nhất trên val') -----
    best_weights = run_dir / "weights" / "best.pt"
    if best_weights.exists():
        results[BASELINE_NAME] = _val(
            best_weights, BASELINE_NAME, f"Baseline — best.pt (split={split})"
        )
    else:
        print(f"  ⚠ Không tìm thấy {best_weights} — bỏ qua Baseline")

    ranked = []
    # ----- CWA: average Top-K checkpoint tốt nhất trên val' -----
    if Config.USE_CWA:
        ranked = rank_checkpoints(run_dir)
        if not ranked:
            print("  ⚠ Không tìm thấy checkpoint epoch nào để average — bỏ qua CWA")
            print("    (cần train với USE_CWA=True để lưu Top-K checkpoint)")
        else:
            print(f"\n  Checkpoint khả dụng (rank theo fitness val'): "
                  f"{[(p.name, round(f, 4)) for p, f, _ in ranked]}")
            mosaic_closed = resolve_mosaic_closed(ranked)

            # Baseline K=1: raw FP32, không average, không đụng BN.
            if Config.EVAL_TOP1_BASELINE:
                results[TOP1_NAME] = _val(
                    ranked[0][0], TOP1_NAME,
                    f"Top-1 raw checkpoint — K=1, không average (split={split})",
                )

            for k in sorted({int(k) for k in Config.TOP_K_VALUES}):
                if k > len(ranked):
                    print(f"  ⚠ Top-{k}: chỉ có {len(ranked)} checkpoint — bỏ qua")
                    continue
                name = cwa_name(k)
                avg_path = run_dir / "weights" / f"cwa_top{k}_avg.pt"
                try:
                    average_checkpoints([p for p, _, _ in ranked[:k]], avg_path)
                    # CRITICAL: BN running stats bị giữ nguyên khi average → phải
                    # re-estimate trên train trước khi val.
                    if Config.USE_BN_UPDATE:
                        update_bn_stats(avg_path, data, mosaic_closed=mosaic_closed)
                    results[name] = _val(
                        avg_path, name, f"CWA — Top-{k} average (split={split})"
                    )
                finally:
                    # Average checkpoint chỉ cần tồn tại trong lúc model.val().
                    # Metrics đã nằm trong RAM nên xóa ngay để giảm disk peak.
                    if Config.DELETE_CHECKPOINTS_AFTER_RUN and avg_path.exists():
                        avg_path.unlink()
                        print(f"      ✓ Đã xóa averaged checkpoint tạm: {avg_path.name}")

    # Dọn thư mục eval_plots rỗng khi SAVE_EVAL_PLOTS=False
    if not plots and eval_root.exists():
        shutil.rmtree(eval_root, ignore_errors=True)

    if not results:
        print("  ✗ Không có strategy nào được đánh giá")
        return {"strategies": {}, "checkpoints": ranking_table(ranked), "excel": None}

    # ----- Bảng so sánh -----
    print("\n" + "=" * 70)
    print(f" STRATEGY COMPARISON (split={split or 'default'})")
    print("=" * 70)
    baseline_metrics = results.get(BASELINE_NAME)
    baseline_value = (
        extract_overall_metrics(baseline_metrics)[HEADLINE_METRIC] if baseline_metrics else None
    )
    for name, metrics in results.items():
        m = extract_overall_metrics(metrics)
        delta = ""
        if baseline_value is not None and name != BASELINE_NAME:
            delta = f" | Δ mAP50-95: {m[HEADLINE_METRIC] - baseline_value:+.4f}"
        print(
            f"  {name:<28} mAP50: {m['mAP@0.5']:.4f} | mAP50-95: {m[HEADLINE_METRIC]:.4f} | "
            f"P: {m['Precision']:.4f} | R: {m['Recall']:.4f}{delta}"
        )
    best_name = max(results, key=lambda n: extract_overall_metrics(results[n])[HEADLINE_METRIC])
    print(f"\n🏆 Best strategy ({HEADLINE_METRIC}): {best_name}")

    excel = export_to_excel(
        run_dir, strategy_results=results, data=data, split=split,
        seed=seed, output_path=excel_path, ranked=ranked,
    )
    return {"strategies": results, "checkpoints": ranking_table(ranked), "excel": excel}


def resolve_data_from_run(run_dir):
    """Lấy path data yaml từ args.yaml mà Ultralytics lưu trong run dir."""
    for candidate in (Path(run_dir) / "args.yaml", Path(run_dir) / "logs" / "args.yaml"):
        if candidate.exists():
            return (yaml.safe_load(candidate.read_text()) or {}).get("data")
    return None


def infer_run_dir(weights):
    """Suy run dir từ vị trí weights chuẩn Ultralytics: <run_dir>/weights/x.pt."""
    weights_path = Path(str(weights))
    if weights_path.parent.name == "weights":
        return weights_path.parent.parent
    return None


# ==================== Excel export per-seed ====================

def read_results_csv(run_dir):
    """Đọc results.csv Ultralytics sinh trong run_dir → DataFrame per-epoch."""
    run_dir = Path(run_dir)
    for candidate in (run_dir / "results.csv", run_dir / "logs" / "results.csv"):
        if candidate.exists():
            df = pd.read_csv(candidate)
            # Bản Ultralytics cũ pad khoảng trắng trong header → chuẩn hóa tên cột
            df.columns = [str(c).strip() for c in df.columns]
            return df
    raise FileNotFoundError(f"results.csv not found in run dir: {run_dir}")


def _data_note(data):
    """Chọn note đúng theo cách chuẩn bị data (holdout hay data.yaml gốc)."""
    if "holdout" in os.path.basename(str(data)):
        return INDEPENDENT_VAL_NOTE
    return ORIGINAL_SPLIT_NOTE


def export_to_excel(run_dir, strategy_results=None, data=None, split=None,
                    output_path=None, seed=None, ranked=None):
    """
    Xuất Excel cho MỘT seed:

    - "Summary"    : run info + overall metrics theo strategy (kèm Δ so với
                     Baseline) + per-class AP theo strategy.
    - "PerEpoch"   : toàn bộ results.csv (loss train/val, P, R, mAP, lr...).
    - "Checkpoints": ranking Top-K theo fitness val' (epoch nào được chọn).
    """
    run_dir = str(run_dir)
    data = data or Config.DATA

    per_epoch_df = None
    try:
        per_epoch_df = read_results_csv(run_dir)
    except FileNotFoundError:
        print(f"  ⚠ Không tìm thấy results.csv trong {run_dir} — bỏ qua sheet PerEpoch")

    # ----- Bảng overall + per-class theo strategy -----
    overall_rows, per_class_rows = [], []
    if strategy_results:
        for name, metrics in strategy_results.items():
            overall_rows.append({"Method": name, **extract_overall_metrics(metrics)})
            for row in extract_per_class_metrics(metrics):
                per_class_rows.append({"Method": name, **row})
        summary_source = "model.val() — Ultralytics SegmentMetrics (mask metrics)"

        # Δ so với Baseline để đọc nhanh strategy nào thắng
        baseline = next(
            (r for r in overall_rows if r["Method"] == BASELINE_NAME), None
        ) or overall_rows[0]
        for row in overall_rows:
            for metric in ("mAP@0.5", HEADLINE_METRIC):
                if metric in row and metric in baseline:
                    row[f"Δ {metric} vs S1"] = round(row[metric] - baseline[metric], 6)
    elif per_epoch_df is not None:
        last = per_epoch_df.iloc[-1]
        column_map = {
            "Precision": "metrics/precision(M)",
            "Recall": "metrics/recall(M)",
            "mAP@0.5": "metrics/mAP50(M)",
            HEADLINE_METRIC: "metrics/mAP50-95(M)",
        }
        row = {"Method": "Last epoch (results.csv)"}
        for metric_name, column in column_map.items():
            if column in per_epoch_df.columns:
                row[metric_name] = float(last[column])
        fitness_columns = (
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "metrics/mAP50(M)",
            "metrics/mAP50-95(M)",
        )
        if all(column in per_epoch_df.columns for column in fitness_columns):
            row["Fitness"] = (
                0.1 * float(last["metrics/mAP50(B)"])
                + 0.9 * float(last["metrics/mAP50-95(B)"])
                + 0.1 * float(last["metrics/mAP50(M)"])
                + 0.9 * float(last["metrics/mAP50-95(M)"])
            )
        overall_rows.append(row)
        summary_source = "results.csv (epoch cuối) — chạy `strategies`/`eval` để có per-class AP"
    else:
        raise ValueError("Không có strategy results lẫn results.csv — không thể export Excel")

    # ----- Block Run Info -----
    try:
        import ultralytics
        ultralytics_version = ultralytics.__version__
    except ImportError:
        ultralytics_version = "N/A"

    run_info_rows = [
        ("model", Config.MODEL or "N/A"),
        ("data", str(data)),
        ("val_ratio (tách từ train)", Config.VAL_RATIO),
        ("epochs", Config.EPOCHS),
        ("imgsz", Config.IMGSZ),
        ("batch", Config.BATCH),
        ("device", Config.DEVICE if Config.DEVICE is not None else "auto"),
        # Ưu tiên seed thực sự dùng cho run này (tham số seed), fallback Config
        ("seed", seed if seed is not None else Config.RANDOM_SEED),
        ("cls_loss", Config.LOSS_FUNCTION + (
            f" (γ={Config.FOCAL_GAMMA}, α={Config.FOCAL_ALPHA})"
            if str(Config.LOSS_FUNCTION).lower() == "focal" else ""
        )),
        ("eval_split", split or Config.EVAL_SPLIT or "mặc định theo data.yaml"),
        ("cwa", f"Top-K {Config.TOP_K_VALUES}" if Config.USE_CWA else "OFF"),
        ("averaging", "uniform element-wise mean của raw FP32 weights (KHÔNG EMA)"),
        ("checkpoint_weight_source", "raw (EMA disabled)" if Config.USE_CWA else "Ultralytics default"),
        ("bn_update", (
            f"ON — {Config.BN_UPDATE_BATCHES} batch train, no-grad forward, "
            f"close_mosaic={Config.BN_UPDATE_CLOSE_MOSAIC}"
            if Config.USE_BN_UPDATE else "OFF"
        )),
        (
            "checkpoint_retention",
            "temporary; deleted after evaluation"
            if Config.DELETE_CHECKPOINTS_AFTER_RUN
            else "kept",
        ),
        ("run_dir", run_dir),
        ("summary_source", summary_source),
        ("export_date", datetime.now().isoformat(timespec="seconds")),
        ("ultralytics_version", ultralytics_version),
        ("NOTE data split", _data_note(data)),
    ]

    if output_path is None:
        if seed is not None:
            default_name = f"seed_{seed}_results.xlsx"
        else:
            default_name = "segmentation_results.xlsx"
        # EXCEL_OUTPUT chỉ áp dụng cho lần export đơn lẻ qua CLI; nếu áp dụng cho
        # mọi seed thì các seed sẽ ghi đè lên nhau.
        output_path = (
            Config.EXCEL_OUTPUT if (Config.EXCEL_OUTPUT and seed is None)
            else os.path.join(run_dir, default_name)
        )
    output_parent = os.path.dirname(str(output_path))
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    info_df = pd.DataFrame(run_info_rows, columns=["Parameter", "Value"])
    overall_df = pd.DataFrame(overall_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    checkpoints_df = pd.DataFrame(ranking_table(ranked or []))

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Sheet "Summary": Run Info → Overall per strategy → Per-class per strategy
        info_df.to_excel(writer, sheet_name="Summary", index=False, startrow=0)
        next_row = len(info_df) + 2
        overall_df.to_excel(writer, sheet_name="Summary", index=False, startrow=next_row)
        next_row += len(overall_df) + 2
        if not per_class_df.empty:
            per_class_df.to_excel(writer, sheet_name="Summary", index=False, startrow=next_row)

        if per_epoch_df is not None:
            per_epoch_df.to_excel(writer, sheet_name="PerEpoch", index=False)
        if not checkpoints_df.empty:
            checkpoints_df.to_excel(writer, sheet_name="Checkpoints", index=False)

    print(f"\n  ✓ Excel exported: {output_path}")
    print(f"    - Sheet 'Summary' : {len(overall_rows)} strategy row(s) + "
          f"{len(per_class_rows)} per-class row(s) + run info")
    if per_epoch_df is not None:
        print(f"    - Sheet 'PerEpoch': {len(per_epoch_df)} epochs (từ results.csv)")
    if not checkpoints_df.empty:
        print(f"    - Sheet 'Checkpoints': {len(checkpoints_df)} checkpoint được giữ")
    return output_path


# ==================== Multi-seed summary export ====================

def _mean_std_frame(df, group_columns, metric_columns):
    """mean/std (ddof=1) + chuỗi 'mean ± std' cho từng nhóm."""
    rows = []
    for keys, group in df.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["N seeds"] = int(len(group))
        for column in metric_columns:
            if column not in group.columns:
                continue
            values = group[column].astype(float)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{column} mean"] = round(mean, 6)
            row[f"{column} std"] = round(std, 6)
            row[f"{column} mean±std"] = f"{mean:.4f} ± {std:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def _delta_frame(summary_df, metric_columns, baseline_name):
    """Chênh lệch ghép cặp theo seed so với baseline + số seed thắng."""
    if baseline_name not in set(summary_df["Method"]):
        return pd.DataFrame()

    base = summary_df[summary_df["Method"] == baseline_name].set_index("Seed")
    rows = []
    for name in dict.fromkeys(summary_df["Method"]):
        if name == baseline_name:
            continue
        group = summary_df[summary_df["Method"] == name].set_index("Seed")
        row = {"Method": name, "Baseline": baseline_name, "N seeds": int(len(group))}
        for column in metric_columns:
            if column not in group.columns or column not in base.columns:
                continue
            deltas = (group[column].astype(float) - base[column].astype(float)).dropna()
            if deltas.empty:
                continue
            mean = float(deltas.mean())
            std = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
            row[f"Δ {column} mean"] = round(mean, 6)
            row[f"Δ {column} std"] = round(std, 6)
            if column == HEADLINE_METRIC:
                row["Seeds tốt hơn baseline"] = f"{int((deltas > 0).sum())}/{len(deltas)}"
        rows.append(row)
    return pd.DataFrame(rows)


def export_multi_seed_summary(all_runs_results, output_dir, exp_name=None, extra_info=None):
    """
    Export Excel tổng hợp + chart cho toàn bộ seed của một experiment.

    Sheets (theo thứ tự mở trong Excel):
      1. "MeanStd"           — mean ± std của mọi metric theo strategy (bảng chính).
      2. "DeltaVsBaseline"   — Δ ghép cặp theo seed so với Baseline + tỉ lệ thắng.
      3. "PerSeed"           — mỗi row = 1 seed × 1 strategy.
      4. "PerClass_MeanStd"  — mean ± std AP theo class × strategy.
      5. "PerClass_PerSeed"  — AP per class thô.
      6. "Checkpoints"       — epoch nào được chọn vào Top-K ở từng seed.
      7. "RunInfo"           — cấu hình experiment.

    Args:
        all_runs_results: list[dict] từ train_detector().
        output_dir: thư mục summary (<exp_dir>/summary).
        exp_name: dùng để đặt tên file.
        extra_info: list[(key, value)] ghi vào sheet RunInfo.

    Returns:
        dict {"excel": Path|None, "charts": [Path], "mean_std": DataFrame}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_stem = f"{exp_name}_summary" if exp_name else "summary"
    output_path = output_dir / f"{file_stem}.xlsx"

    summary_rows, per_class_rows, checkpoint_rows = [], [], []
    for run in all_runs_results:
        seed_val = run["seed"]
        run_dir_str = run.get("run_dir", "")
        for strat_name, metrics in (run.get("strategy_results") or {}).items():
            if metrics is None:
                continue
            summary_rows.append({
                "Seed": seed_val,
                "Method": strat_name,
                **extract_overall_metrics(metrics),
                "Run Folder": run_dir_str,
            })
            for row in extract_per_class_metrics(metrics):
                per_class_rows.append({"Seed": seed_val, "Method": strat_name, **row})
        for row in (run.get("checkpoints") or []):
            checkpoint_rows.append({"Seed": seed_val, **row})

    if not summary_rows:
        print("  ⚠ Không có kết quả nào để export multi-seed summary")
        return {"excel": None, "charts": [], "mean_std": pd.DataFrame()}

    summary_df = pd.DataFrame(summary_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    checkpoints_df = pd.DataFrame(checkpoint_rows)

    metric_columns = [c for c in METRIC_COLUMNS if c in summary_df.columns]
    mean_std_df = _mean_std_frame(summary_df, ["Method"], metric_columns)

    baseline_name = (
        BASELINE_NAME if BASELINE_NAME in set(summary_df["Method"])
        else summary_df["Method"].iloc[0]
    )
    delta_df = _delta_frame(summary_df, metric_columns, baseline_name)

    per_class_mean_std = pd.DataFrame()
    if not per_class_df.empty:
        per_class_mean_std = _mean_std_frame(
            per_class_df, ["Method", "Class"],
            ["AP@0.5", "AP@0.5:0.95", "Precision", "Recall"],
        )

    seeds_done = [r["seed"] for r in all_runs_results]
    info_rows = [
        ("experiment", exp_name or "N/A"),
        ("seeds", ", ".join(str(s) for s in seeds_done)),
        ("n_seeds", len(seeds_done)),
        ("model", Config.MODEL or "N/A"),
        ("data", str(Config.DATA)),
        ("epochs", Config.EPOCHS),
        ("imgsz", Config.IMGSZ),
        ("batch", Config.BATCH),
        ("optimizer", f"{Config.OPTIMIZER} (lr0={Config.LR0}, lrf={Config.LRF}, "
                      f"warmup={Config.WARMUP_EPOCHS}, cos_lr={Config.COS_LR})"),
        ("cls_loss", Config.LOSS_FUNCTION),
        ("eval_split", Config.EVAL_SPLIT),
        ("selection_criterion", "Ultralytics segmentation fitness trên val: "
                                "0.1·mAP50(B)+0.9·mAP50-95(B)+0.1·mAP50(M)+0.9·mAP50-95(M)"),
        ("averaging", "uniform element-wise mean của raw FP32 weights (KHÔNG EMA)"),
        ("top_k_values", str(Config.TOP_K_VALUES)),
        ("bn_update", f"{Config.BN_UPDATE_BATCHES} batch train, no-grad forward, "
                      f"close_mosaic={Config.BN_UPDATE_CLOSE_MOSAIC}"
                      if Config.USE_BN_UPDATE else "OFF"),
        ("std", "sample std (ddof=1) trên các seed"),
        ("export_date", datetime.now().isoformat(timespec="seconds")),
    ]
    info_rows.extend(extra_info or [])
    info_df = pd.DataFrame(info_rows, columns=["Parameter", "Value"])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        mean_std_df.to_excel(writer, sheet_name="MeanStd", index=False)
        if not delta_df.empty:
            delta_df.to_excel(writer, sheet_name="DeltaVsBaseline", index=False)
        summary_df.to_excel(writer, sheet_name="PerSeed", index=False)
        if not per_class_mean_std.empty:
            per_class_mean_std.to_excel(writer, sheet_name="PerClass_MeanStd", index=False)
        if not per_class_df.empty:
            per_class_df.to_excel(writer, sheet_name="PerClass_PerSeed", index=False)
        if not checkpoints_df.empty:
            checkpoints_df.to_excel(writer, sheet_name="Checkpoints", index=False)
        info_df.to_excel(writer, sheet_name="RunInfo", index=False)

    print(f"\n  ✓ Multi-seed summary exported: {output_path}")
    print(f"    Seeds             : {seeds_done}")
    print(f"    Sheet 'MeanStd'   : {len(mean_std_df)} strategy (mean ± std trên {len(seeds_done)} seed)")
    print(f"    Sheet 'PerSeed'   : {len(summary_df)} rows")
    if not delta_df.empty:
        print(f"    Sheet 'DeltaVsBaseline': {len(delta_df)} rows (baseline = {baseline_name})")

    written_charts = []
    if Config.MAKE_CHARTS:
        import charts as charts_module

        written_charts = charts_module.build_summary_charts(
            summary_df, per_class_df, output_dir / "charts", baseline_name
        )
        if written_charts:
            print(f"    Charts            : {len(written_charts)} file → {output_dir / 'charts'}")

    return {
        "excel": output_path,
        "charts": written_charts,
        "mean_std": mean_std_df,
        "summary": summary_df,
        "delta": delta_df,
        "baseline": baseline_name,
    }
