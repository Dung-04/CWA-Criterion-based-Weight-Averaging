"""
Evaluation cho CWA - Object Detection (Ultralytics YOLO + Pascal VOC).

- Metrics đọc TRỰC TIẾP từ DetMetrics của Ultralytics (results.box.*:
  map50, map75, map, mp, mr, per-class AP, fitness) — không tự tính lại mAP.
- Baseline (baseline): checkpoint đơn có fitness cao nhất trên val'.
- CWA: UNIFORM ELEMENT-WISE AVERAGE các learnable parameter của Top-K
  checkpoint tốt nhất trên val'. Không có EMA, không có trọng số theo rank —
  mỗi checkpoint đóng góp đúng 1/K.
- Sau khi average: BatchNorm running stats được ước lượng LẠI bằng cách lặp
  qua training data ở chế độ forward-only (torch.no_grad, không optimizer,
  không backward) — xem update_bn_stats().

Phần trình bày kết quả (Excel, chart, dọn thư mục) nằm ở reporting.py.
"""
import json
import math
import re
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from config import Config
from memory import (
    cpu_quota,
    free_memory,
    release_validator,
    release_yolo,
    shutdown_dataloader,
    validator_of,
)
from reporting import (  # re-export để train.py / main.py import từ 1 chỗ
    export_experiment_charts,
    export_experiment_summary,
    export_seed_excel,
    extract_overall_metrics,
    extract_per_class_metrics,
    make_strategy_record,
    print_detection_metrics,
    print_multi_seed_table,
    read_results_csv,
    slugify,
    snapshot_metrics,
)

# File ranking do TopKCheckpointManager (train.py) ghi trong <run_dir>/weights/
RANKING_FILE = "cwa_checkpoints.json"

# Thư mục con trong run dir chứa output của model.val() cho từng strategy.
EVAL_CHARTS_DIRNAME = "charts"


# ==================== model.val() wrapper ====================

def eval_workers():
    """
    Số worker cho dataloader của các lượt eval/BN sau train.

    Mỗi worker giữ ``prefetch_factor`` batch trong RAM (batch×3×imgsz² byte),
    nên WORKERS=24 với batch=128 là ~7 GB hàng đợi cho MỘT dataloader. Train
    chỉ có 1 loader chạy liên tục nên chịu được; còn mỗi seed lại chạy tới 6
    lượt val + 5 lượt BN recalibration, dùng lại đúng con số đó là tự chuốc
    OOM. None = min(WORKERS, 8).

    Các lượt này gọi ``build_dataloader`` trực tiếp (không có ×2 như val loader
    lúc train), nhưng vẫn clamp theo CPU job được cấp vì Ultralytics chỉ chặn
    bằng ``os.cpu_count()`` của cả node.
    """
    workers = Config.EVAL_WORKERS
    if workers is None:
        workers = min(int(Config.WORKERS), 8)
    workers = max(0, int(workers))
    if Config.AUTO_LIMIT_WORKERS:
        workers = min(workers, cpu_quota())
    return workers


def build_val_args(data, split=None, project=None, name=None):
    """
    Map Config → kwargs của model.val().

    `project`/`name` được truyền tường minh để artifact của val (confusion
    matrix, PR/F1 curve) rơi vào ĐÚNG run dir của seed, thay vì rải ra
    runs/detect/val, val2, val3... ở thư mục làm việc.
    """
    val_args = {
        "data": str(data),
        "imgsz": int(Config.IMGSZ),
        "workers": eval_workers(),
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
    return val_args


def evaluate_weights(weights, data, split=None, header=None, project=None, name=None):
    """
    Chạy model.val() với weights đã train, in metrics, trả về MetricsSnapshot.

    Trả SNAPSHOT chứ không phải DetMetrics: DetMetrics giữ ``on_plot`` = bound
    method của validator → ``validator.dataloader`` → worker process + prefetch
    queue của dataloader. Kết quả eval được giữ tới cuối experiment, nên trả
    thẳng DetMetrics đồng nghĩa với việc mọi dataloader của mọi lượt eval đều
    không bao giờ được thu hồi (xem memory.py).
    """
    from ultralytics import YOLO

    if not weights:
        raise ValueError(
            "Cần weights đã train để eval: truyền --weights path/to/best.pt "
            "(hoặc set Config.MODEL trỏ tới weights đã train trên VOC)."
        )

    model = YOLO(str(weights))
    if getattr(model, "task", None) != "detect":
        raise ValueError(
            f"Pipeline này chỉ hỗ trợ object detection, nhưng {weights!s} "
            f"có task={getattr(model, 'task', None)!r}."
        )

    # Model.val() không giữ lại validator ở đâu cả, nên bắt nó qua callback để
    # còn đóng dataloader ngay khi val xong.
    validators = []
    model.add_callback("on_val_end", validators.append)
    metrics = None
    try:
        metrics = model.val(**build_val_args(data, split, project=project, name=name))
        snapshot = snapshot_metrics(metrics)
    finally:
        # validator_of(): phòng khi một bản Ultralytics nào đó không chạy
        # on_val_end — DetMetrics vẫn trỏ ngược về validator qua on_plot.
        for validator in validators + [validator_of(metrics)]:
            release_validator(validator)
        release_yolo(model)
        metrics = None
        del model
        free_memory()

    print_detection_metrics(
        snapshot,
        header=header or f"EVALUATION (split={split or 'default'}) — {Path(str(weights)).name}",
    )
    return snapshot


# ==================== CWA: ranking ====================

def _rank_key(fitness, epoch):
    """
    Khóa sắp xếp: fitness cao trước; hòa thì epoch LỚN hơn (muộn hơn) trước.

    Khớp đúng ngữ nghĩa best.pt của Ultralytics. Trong BaseTrainer:

        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness          # chỉ đổi khi LỚN HƠN HẲN
        ...
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)   # nhưng GHI LẠI khi BẰNG

    Nên khi hòa, best.pt bị ghi đè bằng epoch MUỘN hơn. Rank #1 của ta vì vậy
    cũng phải chọn epoch muộn hơn thì mới trùng epoch với best.pt.
    """
    return (fitness, epoch)


def rank_checkpoints(run_dir):
    """
    Rank các raw checkpoint epoch theo fitness trên val' (giảm dần).

    Nguồn chính: cwa_checkpoints.json (TopKCheckpointManager ghi lúc train).
    Fallback: tự tính fitness = 0.1*mAP50 + 0.9*mAP50-95 từ results.csv
    (khi run dir được copy từ máy khác mà thiếu file json).

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
        fitness_by_epoch = {}
        if "metrics/mAP50(B)" in df.columns and "metrics/mAP50-95(B)" in df.columns:
            for _, row in df.iterrows():
                fitness = (
                    0.1 * float(row["metrics/mAP50(B)"])
                    + 0.9 * float(row["metrics/mAP50-95(B)"])
                )
                if math.isfinite(fitness):
                    fitness_by_epoch[int(row["epoch"])] = fitness
        for path in weights_dir.glob("epoch*.pt"):
            digits = re.sub(r"\D", "", path.stem)
            if not digits:
                continue
            file_epoch = int(digits)
            # Tên file epoch{N}.pt đánh số 0-based, cột epoch results.csv 1-based
            fitness = fitness_by_epoch.get(file_epoch + 1, fitness_by_epoch.get(file_epoch))
            if fitness is not None:
                records.append((path, float(fitness), file_epoch))

    records.sort(key=lambda r: _rank_key(r[1], r[2]), reverse=True)
    return records


def ranking_to_records(ranked):
    """[(path, fitness, epoch)] → list dict để đưa vào Excel/JSON."""
    return [
        {"file": path.name, "fitness": float(fitness), "epoch": int(epoch)}
        for path, fitness, epoch in ranked
    ]


# ==================== CWA: element-wise weight averaging ====================

def average_checkpoints(ckpt_paths, output_path, verbose=True):
    """
    UNIFORM ELEMENT-WISE AVERAGE các learnable parameter của K checkpoint.

    Công thức, áp dụng độc lập cho từng phần tử của từng tensor tham số θ:

        θ_avg = (1/K) * Σ_{i=1..K} θ_i

    Đây KHÔNG phải EMA: không có hệ số decay, không phụ thuộc thứ tự, mỗi
    checkpoint đóng góp đúng 1/K bất kể rank hay epoch.

    Cái gì được average / không:
      - ĐƯỢC : mọi learnable parameter dạng float — conv/linear weight, bias,
               gamma & beta của BatchNorm (đều nằm trong named_parameters()).
      - KHÔNG: buffer và state không learnable — BN running_mean/running_var/
               num_batches_tracked, anchors, stride, cache... Trung bình cộng
               các đại lượng thống kê này không có ý nghĩa; chúng được giữ từ
               checkpoint TỐT NHẤT rồi ước lượng lại bằng update_bn_stats().

    Phép cộng thực hiện ở float64 rồi mới hạ về dtype gốc, nên kết quả không
    phụ thuộc thứ tự cộng và không tích lũy sai số làm tròn theo K.

    Input chỉ nhận checkpoint có ``weight_source='raw'``. Output giữ FP32,
    không chứa EMA/optimizer, load lại bằng YOLO(path) như checkpoint thường.
    """
    import torch

    ckpt_paths = [Path(p) for p in ckpt_paths]
    if not ckpt_paths:
        raise ValueError("average_checkpoints() cần ít nhất 1 checkpoint")
    resolved = [p.resolve() for p in ckpt_paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Danh sách checkpoint bị trùng: {[p.name for p in ckpt_paths]}")

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

    # Mọi checkpoint phải cùng kiến trúc — nếu lệch key/shape thì average sai
    # về mặt ngữ nghĩa, phải dừng chứ không được im lặng bỏ qua.
    reference_keys = set(state_dicts[0])
    for path, sd in zip(ckpt_paths[1:], state_dicts[1:]):
        if set(sd) != reference_keys:
            missing = sorted(reference_keys - set(sd))[:5]
            extra = sorted(set(sd) - reference_keys)[:5]
            raise ValueError(
                f"State dict của {path.name} không khớp {ckpt_paths[0].name}: "
                f"thiếu {missing}, thừa {extra}"
            )
        for key in reference_keys:
            if sd[key].shape != state_dicts[0][key].shape:
                raise ValueError(
                    f"Shape lệch ở key {key!r}: {ckpt_paths[0].name}"
                    f"{tuple(state_dicts[0][key].shape)} vs {path.name}{tuple(sd[key].shape)}"
                )

    k = len(state_dicts)
    averaged = copied = 0
    avg_state = {}
    for key, ref_tensor in state_dicts[0].items():
        is_learnable_float = key in parameter_keys and ref_tensor.dtype.is_floating_point
        if k == 1 or not is_learnable_float:
            # K=1 → average của 1 phần tử chính là chính nó (clone để an toàn).
            avg_state[key] = ref_tensor.detach().clone()
            copied += 0 if k == 1 else 1
            continue
        # Cộng ở float64 → chia K. Uniform, không trọng số, không EMA.
        accumulator = ref_tensor.detach().to(torch.float64).clone()
        for sd in state_dicts[1:]:
            accumulator += sd[key].detach().to(torch.float64)
        accumulator /= float(k)
        avg_state[key] = accumulator.to(ref_tensor.dtype)
        averaged += 1

    merged_module = modules[0]
    merged_module.load_state_dict(avg_state, strict=True)

    source_epochs = [int(ckpt.get("epoch", -1)) for ckpt in ckpts]
    torch.save(
        {
            # FP32 giảm sai số làm tròn so với FP16 khi cộng/chia K tensors.
            "model": merged_module.float(),
            "ema": None,
            "optimizer": None,
            "epoch": -1,
            "train_args": ckpts[0].get("train_args", {}),
            "date": datetime.now().isoformat(timespec="seconds"),
            "weight_source": "raw_topk_average",
            "average_k": k,
            "average_type": "uniform_elementwise_mean",
            "source_checkpoints": [p.name for p in ckpt_paths],
            "source_epochs": source_epochs,
            "bn_recalibrated": False,
        },
        str(output_path),
    )
    if verbose:
        print(
            f"      ✓ Averaged K={k} checkpoint (epoch {source_epochs}) — "
            f"{averaged} tensor tham số lấy trung bình 1/{k}, "
            f"{copied} buffer/non-learnable giữ từ checkpoint tốt nhất"
        )

    # K model FP32 + K state_dict + accumulator float64 đang nằm trong RAM;
    # bỏ ngay chứ không chờ hết hàm, bước kế tiếp (BN recal) lại ngốn tiếp.
    del ckpts, modules, state_dicts, avg_state, merged_module
    free_memory()
    return Path(output_path)


# ==================== CWA: BatchNorm recalibration ====================

def update_bn_stats(weights_path, data_yaml, num_batches=None, device=None):
    """
    Ước lượng LẠI BatchNorm running statistics của model đã average.

    Vì sao cần: average_checkpoints() giữ nguyên running_mean/running_var từ
    checkpoint tốt nhất (không average vì đó là thống kê quần thể) nhưng weights
    của các layer phía trước BN đã đổi → phân phối activation không còn khớp
    thống kê cũ. Không hiệu chỉnh thì mAP của CWA tụt do BN lệch.

    Cách làm (tương đương torch.optim.swa_utils.update_bn):
      1. reset_running_stats() + momentum = None → BN dùng CUMULATIVE MOVING
         AVERAGE, tức trung bình chính xác trên toàn bộ batch đã đi qua,
         không phụ thuộc thứ tự batch và không có hệ số quán tính.
      2. LẶP QUA TRAINING DATA ở chế độ forward-only:
         - toàn bộ chạy trong torch.no_grad() → KHÔNG dựng graph, KHÔNG backward
         - KHÔNG có optimizer, KHÔNG optimizer.step()
         - model.eval() cho toàn mạng (tắt dropout/nhánh train), riêng các
           module BN bật train() để chúng tích lũy mean/var
         - kiểm tra lại sau vòng lặp: mọi parameter.grad phải là None
      3. Chỉ running_mean/running_var/num_batches_tracked thay đổi; learnable
         parameters giữ nguyên hệt giá trị vừa average.

    Dữ liệu dùng: SPLIT TRAIN của data_yaml (train' sau khi đã tách val').
    Tuyệt đối không đụng val' hay test.

    Args:
        weights_path: file .pt đã average (do `average_checkpoints` sinh ra).
        data_yaml: path data yaml (holdout hoặc gốc).
        num_batches: số batch forward (mặc định Config.BN_UPDATE_BATCHES).
        device: None = auto GPU nếu có.

    Returns:
        weights_path (đã ghi đè với BN stats mới).
    """
    import torch
    from torch.nn.modules.batchnorm import _BatchNorm
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    weights_path = Path(weights_path)
    num_batches = int(num_batches or Config.BN_UPDATE_BATCHES)

    yolo = YOLO(str(weights_path))
    model = yolo.model
    bn_modules = [m for m in model.modules() if isinstance(m, _BatchNorm)]
    if not bn_modules:
        print("      (Không tìm thấy BN layer nào — bỏ qua BN recalibration)")
        return weights_path

    device_obj = torch.device(
        device
        or (f"cuda:{Config.DEVICE}" if isinstance(Config.DEVICE, str) and Config.DEVICE.isdigit()
            else "cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.float().to(device_obj)
    # Không cần gradient ở bất kỳ đâu trong hàm này.
    model.requires_grad_(False)
    stride = int(max(model.stride)) if hasattr(model, "stride") else 32

    # Dataloader kiểu Ultralytics trên split TRAIN.
    # mode="train" giữ nguyên augmentation của lúc train: BN stats khi train
    # vốn được tích lũy trên phân phối đã augment, nên muốn tái lập đúng
    # estimator đó thì phải forward trên cùng phân phối.
    data = check_det_dataset(str(data_yaml))
    mode = "train" if Config.BN_UPDATE_AUGMENT else "val"
    batch = int(Config.BATCH) if int(Config.BATCH) > 0 else 16
    bn_overrides = {
        "imgsz": int(Config.IMGSZ),
        "task": "detect",
        "mode": "train",
        "mixup": float(Config.MIXUP),
        "copy_paste": float(Config.COPY_PASTE),
    }
    bn_overrides.update(Config.EXTRA_TRAIN_ARGS or {})
    bn_overrides["task"] = "detect"
    cfg = get_cfg(overrides=bn_overrides)
    dataset = build_yolo_dataset(
        cfg, data["train"], batch, data, mode=mode, rect=False, stride=stride
    )
    loader = build_dataloader(dataset, batch, workers=eval_workers(), shuffle=True, rank=-1)
    batches_per_epoch = max(len(loader), 1)
    if num_batches > batches_per_epoch:
        print(
            f"      ⚠ BN_UPDATE_BATCHES={num_batches} > số batch/epoch "
            f"({batches_per_epoch}) → một phần ảnh sẽ được duyệt lại"
        )

    # momentum=None ⇒ cumulative moving average trên toàn bộ batch đã thấy.
    saved_momentum = [(m, m.momentum) for m in bn_modules]
    was_training = model.training
    seen_images = seen_batches = 0
    try:
        for module in bn_modules:
            module.reset_running_stats()
            module.momentum = None

        model.eval()
        for module in bn_modules:
            module.train()

        # Autocast khớp precision lúc train. BN chỉ tích lũy mean/var nên không
        # nhạy với precision, nhưng giữ cho nhất quán và nhanh hơn trên GPU.
        if Config.BN_UPDATE_AMP and device_obj.type == "cuda":
            autocast_ctx = torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            )
        else:
            autocast_ctx = torch.autocast(device_type=device_obj.type, enabled=False)

        print(
            f"      BN recalibration: {num_batches} batch × {batch} ảnh trên split TRAIN "
            f"(mode={mode}, device={device_obj}, no_grad, không optimizer)..."
        )
        with torch.no_grad(), autocast_ctx:
            for i, batch_data in enumerate(loader):
                if i >= num_batches:
                    break
                img = batch_data["img"].to(device_obj, non_blocking=True).float() / 255.0
                model(img)
                seen_images += int(img.shape[0])
                seen_batches += 1
    finally:
        for module, momentum in saved_momentum:
            module.momentum = momentum
        model.train(was_training)
        # Vòng lặp thoát sớm bằng `break` nên iterator vẫn còn sống cùng toàn bộ
        # worker process + prefetch queue. Đóng tay ngay: hàm này được gọi tới 5
        # lần mỗi seed, để GC lo thì RAM chồng lên nhau.
        shutdown_dataloader(loader)
        loader = dataset = None
        free_memory()

    # Bằng chứng "chỉ forward, không cập nhật gradient": không parameter nào có .grad
    with_grad = [name for name, p in model.named_parameters() if p.grad is not None]
    if with_grad:
        raise RuntimeError(
            "BN recalibration phải là forward-only nhưng có gradient được tích lũy: "
            f"{with_grad[:5]}"
        )

    # Chỉ thay BN running statistics; learnable parameters vẫn là bản average FP32.
    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    ckpt["model"] = model.float().cpu()
    # attempt_load_one_weight() ưu tiên ckpt["ema"] hơn ckpt["model"]; nếu file
    # nguồn còn field 'ema' thì model vừa hiệu chỉnh BN sẽ bị bỏ qua khi load.
    ckpt["ema"] = None
    ckpt["bn_recalibrated"] = True
    ckpt["bn_update_images"] = seen_images
    ckpt["bn_update_batches"] = seen_batches
    torch.save(ckpt, str(weights_path))
    print(f"      ✓ BN stats updated trên {seen_images} ảnh train: {weights_path.name}")

    del ckpt, model, yolo, bn_modules, saved_momentum
    free_memory()
    return weights_path


# ==================== Strategy orchestration ====================

def _eval_record(weights, data, split, run_dir, name, short, kind, k=None,
                 epochs=None, bn_update=False):
    """Eval 1 checkpoint và đóng gói thành strategy record (chart của val vào run dir)."""
    metrics = evaluate_weights(
        weights,
        data,
        split,
        header=f"{name} (split={split or 'default'})",
        project=Path(run_dir) / EVAL_CHARTS_DIRNAME,
        name=f"test_{slugify(short)}",
    )
    return make_strategy_record(name, short, kind, metrics, k=k, epochs=epochs, bn_update=bn_update)


def run_method_evaluation(run_dir, data=None, split=None, seed=None, export_excel=True):
    """
    Đánh giá trên split báo cáo (mặc định Config.EVAL_SPLIT='test' — VOC2007 test):

      - Baseline          : checkpoint đơn tốt nhất trên val'
      - Baseline + BN     : cùng checkpoint đó nhưng ĐÃ BN-recalibrate (control
                              để tách phần cải thiện do averaging khỏi phần do
                              hiệu chỉnh BN) — bật/tắt bằng Config.BN_UPDATE_CONTROL
      - CWA (Top-K)  : average Top-K checkpoint, K ∈ Config.TOP_K_VALUES

    Returns:
        dict {"strategy_records": [...], "ranking": [...], "excel": path|None}
    """
    run_dir = Path(run_dir)
    data = data or resolve_data_from_run(run_dir) or Config.DATA
    split = split or Config.EVAL_SPLIT
    weights_dir = run_dir / "weights"

    print("\n" + "=" * 70)
    print(f" STRATEGY EVALUATION (split={split or 'default'}, data={data})")
    print("=" * 70)

    records = []
    temp_files = []
    ranked = rank_checkpoints(run_dir) if Config.USE_CWA else []
    if ranked:
        print("  Ranking checkpoint theo fitness trên val' (cao → thấp):")
        for rank, (path, fitness, epoch) in enumerate(ranked, start=1):
            print(f"    #{rank}  epoch {epoch:<4} fitness {fitness:.6f}  ({path.name})")

    try:
        # ----- Baseline: checkpoint đơn tốt nhất trên val' -----
        best_weights = weights_dir / "best.pt"
        baseline_epochs = []
        if ranked and Config.BASELINE_FROM_RAW_TOPK:
            # Dùng rank #1 (raw FP32) thay vì best.pt (Ultralytics lưu FP16):
            # cùng epoch, cùng weights, nhưng cùng precision và cùng đường
            # load/eval với CWA → so sánh mới thực sự apples-to-apples.
            baseline_path, _, baseline_epoch = ranked[0]
            baseline_epochs = [baseline_epoch]
            print(f"\n  Baseline dùng rank #1 raw FP32 (epoch {baseline_epoch}) "
                  f"— cùng epoch với best.pt, tránh lệch precision FP16 khi so sánh")
        elif best_weights.exists():
            baseline_path = best_weights
        else:
            baseline_path = None
            print(f"  ⚠ Không tìm thấy {best_weights} — bỏ qua Baseline")

        if baseline_path is not None:
            records.append(_eval_record(
                baseline_path, data, split, run_dir,
                name="Baseline (best ckpt)", short="Best", kind="baseline",
                k=1, epochs=baseline_epochs,
            ))

            # ----- Control: baseline + BN recalibration -----
            # CWA được BN-recalibrate; nếu không có control này thì không
            # thể biết phần cải thiện đến từ averaging hay từ hiệu chỉnh BN.
            if Config.USE_CWA and Config.USE_BN_UPDATE and Config.BN_UPDATE_CONTROL and ranked:
                bn_ctrl_path = weights_dir / "baseline_bn_control.pt"
                temp_files.append(bn_ctrl_path)
                shutil.copyfile(baseline_path, bn_ctrl_path)
                update_bn_stats(bn_ctrl_path, data)
                records.append(_eval_record(
                    bn_ctrl_path, data, split, run_dir,
                    name="Baseline + BN recal", short="Best+BN", kind="baseline_bn",
                    k=1, epochs=baseline_epochs, bn_update=True,
                ))

        # ----- CWA: average Top-K checkpoint tốt nhất trên val' -----
        if Config.USE_CWA:
            if not ranked:
                print("  ⚠ Không tìm thấy checkpoint epoch nào để average — bỏ qua CWA")
                print("    (cần train với USE_CWA=True để lưu Top-K checkpoint)")
            for k in sorted(int(k) for k in Config.TOP_K_VALUES):
                if k > len(ranked):
                    print(f"  ⚠ Top-{k}: chỉ có {len(ranked)} checkpoint — bỏ qua")
                    continue
                avg_path = weights_dir / f"cwa_top{k}_avg.pt"
                temp_files.append(avg_path)
                selected = ranked[:k]
                average_checkpoints([p for p, _, _ in selected], avg_path)
                if Config.USE_BN_UPDATE:
                    # Re-estimate BN buffers trên train — tuyệt đối không dùng test.
                    update_bn_stats(avg_path, data)
                records.append(_eval_record(
                    avg_path, data, split, run_dir,
                    name=f"CWA (Top-{k})", short=f"Top-{k}", kind="topk_avg",
                    k=k, epochs=[e for _, _, e in selected],
                    bn_update=bool(Config.USE_BN_UPDATE),
                ))
    finally:
        # Checkpoint average/control chỉ cần tồn tại trong lúc model.val().
        if Config.DELETE_CHECKPOINTS_AFTER_RUN:
            for path in temp_files:
                if path.exists():
                    path.unlink()

    if not records:
        print("  ✗ Không có strategy nào được đánh giá")
        return {"strategy_records": [], "ranking": ranking_to_records(ranked), "excel": None}

    # ----- Bảng so sánh -----
    print("\n" + "=" * 78)
    print(f" STRATEGY COMPARISON (split={split or 'default'}, seed={seed})")
    print("=" * 78)
    baseline_overall = extract_overall_metrics(records[0]["metrics"])
    for record in records:
        m = extract_overall_metrics(record["metrics"])
        delta = m["mAP@0.5:0.95"] - baseline_overall["mAP@0.5:0.95"]
        delta_str = "  (baseline)" if record is records[0] else f"  Δ {delta:+.4f}"
        print(
            f"  {record['name']:<26} mAP50: {m['mAP@0.5']:.4f} | "
            f"mAP50-95: {m['mAP@0.5:0.95']:.4f} | P: {m['Precision']:.4f} | "
            f"R: {m['Recall']:.4f}{delta_str}"
        )
    print(
        "\n  Các K ở trên đã được định trước trong Config.TOP_K_VALUES. "
        "Split báo cáo KHÔNG được dùng để chọn K."
    )

    excel_path = None
    if export_excel:
        excel_path = export_seed_excel(
            run_dir,
            strategy_results=records,
            data=data,
            split=split,
            seed=seed,
            ranking=ranking_to_records(ranked),
        )

    return {
        "strategy_records": records,
        "ranking": ranking_to_records(ranked),
        "excel": excel_path,
    }


def resolve_data_from_run(run_dir):
    """Lấy path data yaml từ args.yaml mà Ultralytics lưu trong run dir."""
    args_path = Path(run_dir) / "args.yaml"
    if args_path.exists():
        return (yaml.safe_load(args_path.read_text()) or {}).get("data")
    return None


def infer_run_dir(weights):
    """Suy run dir từ vị trí weights chuẩn Ultralytics: <run_dir>/weights/x.pt."""
    weights_path = Path(str(weights))
    if weights_path.parent.name == "weights":
        return weights_path.parent.parent
    return None


def export_to_excel(run_dir, strategy_results=None, data=None, split=None,
                    output_path=None, seed=None):
    """Wrapper giữ tương thích cho `main.py eval` / `main.py export`."""
    return export_seed_excel(
        run_dir,
        strategy_results=strategy_results,
        data=data,
        split=split,
        output_path=output_path or Config.EXCEL_OUTPUT,
        seed=seed,
    )


__all__ = [
    "RANKING_FILE",
    "average_checkpoints",
    "build_val_args",
    "eval_workers",
    "evaluate_weights",
    "export_experiment_charts",
    "export_experiment_summary",
    "export_seed_excel",
    "export_to_excel",
    "extract_overall_metrics",
    "extract_per_class_metrics",
    "infer_run_dir",
    "print_detection_metrics",
    "print_multi_seed_table",
    "rank_checkpoints",
    "ranking_to_records",
    "read_results_csv",
    "resolve_data_from_run",
    "run_method_evaluation",
    "snapshot_metrics",
    "update_bn_stats",
]
