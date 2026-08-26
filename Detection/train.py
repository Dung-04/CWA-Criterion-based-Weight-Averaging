"""
Train Ultralytics YOLO cho object detection (Pascal VOC).

Toàn bộ training loop giao cho Ultralytics (model.train): LR schedule,
augmentation, early stopping, best.pt/last.pt, results.csv đều do Ultralytics
quản lý trong run dir. Phần thêm vào cho CWA:

- save_period=1 để Ultralytics lưu checkpoint mỗi epoch, NHƯNG
- TopKCheckpointManager (callback on_model_save) ghi đè checkpoint đó bằng RAW
  FP32 model rồi prune NGAY nếu nó nằm ngoài Top-K theo fitness trên val' →
  disk chỉ giữ đúng K checkpoint cần thiết (+ best.pt/last.pt), không phình
  theo số epoch.

Fitness được lấy trực tiếp từ ``trainer.fitness`` — đúng cùng đại lượng mà
Ultralytics dùng cho best.pt và early stopping. EMA được TẮT để validation,
fitness, best.pt và các checkpoint epoch đều dùng raw model weights.

Kết quả cuối cùng của một experiment (xem reporting.py để biết cây thư mục):
mỗi seed một thư mục chỉ chứa Excel + charts, cộng một SUMMARY.xlsx tổng hợp
mean ± std trên tất cả seed. Không giữ lại checkpoint.
"""
import json
import math
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from config import Config
from dataset import prepare_dataset
from evaluate import (
    RANKING_FILE,
    export_experiment_charts,
    export_experiment_summary,
    print_detection_metrics,
    print_multi_seed_table,
    run_method_evaluation,
)
from losses import assert_cls_loss_installed, install_cls_loss
from memory import (
    cpu_quota,
    free_memory,
    memory_report,
    release_yolo,
    training_prefetch_bytes,
)
from reporting import tidy_run_dir


def unwrap_ultralytics_model(model):
    """
    Lấy base model tương thích cả Ultralytics cũ và mới.

    - Bản mới dùng ``unwrap_model`` (hỗ trợ torch.compile + DP/DDP).
    - ultralytics==8.3.152 dùng tên cũ ``de_parallel``.
    - Fallback cuối giữ đúng logic unwrap chính thức để code không phụ thuộc
      cứng vào tên helper nội bộ của Ultralytics.
    """
    try:
        from ultralytics.utils.torch_utils import unwrap_model
    except ImportError:
        try:
            from ultralytics.utils.torch_utils import de_parallel
        except ImportError:
            from torch import nn

            while True:
                if hasattr(model, "_orig_mod") and isinstance(model._orig_mod, nn.Module):
                    model = model._orig_mod
                elif hasattr(model, "module") and isinstance(model.module, nn.Module):
                    model = model.module
                else:
                    return model
        else:
            return de_parallel(model)
    else:
        return unwrap_model(model)


def materialize_inference_tensors(module):
    """
    Thay mọi *inference tensor* trong ``module`` bằng clone thường (in-place).

    Vì sao cần: validator của Ultralytics chạy dưới ``torch.inference_mode()``
    (``BaseValidator.__call__`` được decorate bằng ``smart_inference_mode``) và
    trong lúc train nó gọi ``model.half()`` rồi ``model.float()`` trên chính
    module validation này. ``nn.Module._apply`` THAY THẾ buffer bằng tensor mới::

        self._buffers[key] = fn(buf)

    nên các buffer float của BatchNorm (running_mean/running_var) được TẠO RA
    bên trong inference_mode ⇒ chúng trở thành inference tensor. Sang epoch sau,
    ``load_state_dict`` copy in-place vào đúng các buffer đó nhưng ở NGOÀI
    inference_mode, và PyTorch chặn::

        While copying the parameter named "model.23.cv3.2.0.1.bn.running_mean"
        ... Inplace update to inference tensor outside InferenceMode is not
        allowed. You can make a clone to get a normal tensor before doing
        inplace update.

    Parameter không dính lỗi này vì ``_apply`` đi đường ``param.data = ...``
    (giữ nguyên object Parameter bên ngoài), còn ``num_batches_tracked`` là
    int64 nên ``.half()/.float()`` trả về chính nó — khớp đúng với việc log lỗi
    chỉ liệt kê running_mean/running_var.

    Clone (đúng cách PyTorch gợi ý trong thông báo lỗi) trả về tensor thường,
    giữ nguyên giá trị/dtype/device/shape, nên phép copy in-place sau đó hợp lệ
    trở lại. Chỉ tensor nào thực sự là inference tensor mới bị thay, module
    không có thì hàm là no-op.

    CHÚ Ý: hàm tạo object ``Parameter`` MỚI, nên chỉ dùng cho module validation.
    Không gọi trên ``trainer.model`` — optimizer giữ tham chiếu tới các
    Parameter cũ, thay object sẽ khiến optimizer update nhầm tensor.

    Returns:
        Số tensor đã được thay bằng clone.
    """
    import torch

    replaced = 0
    with torch.no_grad():
        for submodule in module.modules():
            for name, buf in list(submodule._buffers.items()):
                if buf is not None and buf.is_inference():
                    submodule._buffers[name] = buf.clone()
                    replaced += 1
            for name, param in list(submodule._parameters.items()):
                if param is not None and param.is_inference():
                    submodule._parameters[name] = torch.nn.Parameter(
                        param.clone(), requires_grad=param.requires_grad
                    )
                    replaced += 1
    return replaced


class TopKCheckpointManager:
    """
    Quản lý raw-model checkpoint cho CWA.

    ``on_train_start`` tắt EMA smoothing. ``on_train_epoch_end`` copy 1:1 raw
    state sang module validation riêng, nhờ đó fitness, early stopping và
    best.pt dùng raw weights mà không chạy inference trên training module.

    ``on_model_save`` ghi đè epoch checkpoint bằng raw FP32 model của epoch hiện
    tại, ghi nhận fitness trên val' và xóa checkpoint tệ nhất nếu vượt
    KEEP_TOP_K_CHECKPOINTS.

    Ranking được ghi ra <run_dir>/weights/cwa_checkpoints.json để
    evaluate.rank_checkpoints() dùng lại khi average Top-K (CWA).
    Không đụng tới best.pt / last.pt của Ultralytics.
    """

    def __init__(self, keep_top_k):
        self.keep_top_k = int(keep_top_k)
        self.records = {}  # filename -> {"epoch": int, "fitness": float, "weight_source": "raw"}

    @staticmethod
    def _quality_key(info):
        """
        Khóa "checkpoint tốt đến đâu": fitness cao hơn thì tốt hơn; hòa thì
        epoch MUỘN hơn được coi là tốt hơn.

        Khớp đúng ngữ nghĩa best.pt của Ultralytics: `best_fitness` chỉ đổi khi
        fitness LỚN HƠN HẲN, nhưng best.pt lại được ghi lại mỗi khi
        `best_fitness == fitness` — tức là khi hòa, epoch muộn hơn thắng.
        Giống hệt evaluate._rank_key() nên rank #1 luôn cùng epoch với best.pt.
        """
        return (info["fitness"], info["epoch"])

    @staticmethod
    def on_train_start(trainer):
        """
        Tắt EMA smoothing nhưng giữ model validation là một module riêng.

        Không được trỏ ``ema.ema`` trực tiếp vào ``trainer.model``: validator
        chạy dưới torch.inference_mode() và YOLO head có thể cache inference
        tensors vào module, khiến backward của epoch sau bị RuntimeError.
        """
        if getattr(trainer, "ema", None) is None:
            raise RuntimeError("Ultralytics trainer chưa khởi tạo ModelEMA.")

        # ModelEMA.update() trở thành no-op. ema.ema vẫn là deepcopy riêng được
        # Ultralytics tạo lúc setup; ta chỉ đồng bộ raw state_dict trước mỗi val.
        trainer.ema.enabled = False
        TopKCheckpointManager.sync_raw_validation_model(trainer)
        print(
            "  CWA: EMA smoothing disabled — validation dùng bản sao "
            "đồng bộ chính xác RAW weights"
        )

    @staticmethod
    def sync_raw_validation_model(trainer):
        """
        Copy chính xác raw weights/buffers sang model validation riêng.

        Đây là phép copy 1:1 tại cùng epoch, không phải exponential moving
        average và không tạo learnable parameter mới.
        """
        import torch

        raw_model = unwrap_ultralytics_model(trainer.model)
        validation_model = unwrap_ultralytics_model(trainer.ema.ema)

        # Phòng trường hợp một phiên bản/custom trainer đã alias hai object.
        if validation_model is raw_model:
            validation_model = deepcopy(raw_model).eval()
            validation_model.requires_grad_(False)
            trainer.ema.ema = validation_model

        # Sau lần validation trước, các buffer BN của module này đã bị
        # model.half()/model.float() của validator tạo lại BÊN TRONG
        # torch.inference_mode() ⇒ là inference tensor và không cho copy in-place
        # ở ngoài inference_mode. Đổi chúng thành tensor thường trước khi copy.
        materialize_inference_tensors(validation_model)

        with torch.no_grad():
            # Validator của Ultralytics gọi model.half() khi val trong lúc train,
            # nên module này có thể đang là FP16; load_state_dict tự cast, còn
            # checkpoint Top-K vẫn được lấy từ trainer.model ở FP32.
            validation_model.load_state_dict(raw_model.state_dict(), strict=True)
        validation_model.eval()

    @staticmethod
    def on_train_epoch_end(trainer):
        """Đồng bộ raw snapshot sau optimizer step cuối, ngay trước validation."""
        TopKCheckpointManager.sync_raw_validation_model(trainer)

    def on_model_save(self, trainer):
        weights_dir = Path(trainer.save_dir) / "weights"
        ranking_path = weights_dir / RANKING_FILE

        # Resume-safe: chỉ nhận ranking raw mới; run cũ dùng EMA/val_loss phải
        # train lại vì checkpoint phù hợp có thể đã bị prune.
        if not self.records and ranking_path.exists():
            saved = json.loads(ranking_path.read_text(encoding="utf-8"))
            if any(
                "fitness" not in info or info.get("weight_source") != "raw"
                for info in saved.values()
            ):
                raise RuntimeError(
                    f"{ranking_path} không phải ranking raw-weight hiện tại. "
                    "Hãy dùng run mới để tạo Top-K raw checkpoints."
                )
            self.records = {
                name: {
                    "epoch": int(info["epoch"]),
                    "fitness": float(info["fitness"]),
                    "weight_source": "raw",
                }
                for name, info in saved.items()
                if (weights_dir / name).exists()
            }

        # Ultralytics đã validate trước callback này; trainer.fitness chính là
        # criterion dùng cho best.pt và EarlyStopping.
        fitness = getattr(trainer, "fitness", None)
        try:
            fitness = float(fitness)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Ultralytics không cung cấp trainer.fitness hợp lệ.") from exc
        if not math.isfinite(fitness):
            raise RuntimeError(f"trainer.fitness không hữu hạn tại epoch {trainer.epoch}: {fitness}")

        # save_period=1 tạo đúng epoch{trainer.epoch}.pt trước on_model_save.
        ckpt = weights_dir / f"epoch{int(trainer.epoch)}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Checkpoint vừa lưu không tồn tại: {ckpt}. "
                "CWA cần save=True và save_period=1."
            )

        # Ghi đè epoch checkpoint mặc định (field 'ema', FP16) bằng raw FP32
        # model rõ ràng trong field 'model'. Nguồn là trainer.model — luôn FP32
        # và chưa từng chạy dưới inference_mode nên deepcopy an toàn. Không lưu
        # optimizer vì file này chỉ dùng cho post-training averaging.
        import torch
        import ultralytics

        raw_model = deepcopy(unwrap_ultralytics_model(trainer.model)).float().cpu()
        # Bỏ criterion khỏi checkpoint (giống strip_optimizer của Ultralytics):
        # nó chỉ dùng lúc train, lại giữ tham chiếu ngược về model và kéo theo
        # class custom (FocalBCE) vào file pickle.
        raw_model.criterion = None
        raw_ckpt = {
            "epoch": int(trainer.epoch),
            "best_fitness": float(getattr(trainer, "best_fitness", fitness)),
            "model": raw_model,
            "ema": None,
            "updates": 0,
            "optimizer": None,
            "train_args": vars(trainer.args),
            "train_metrics": {**getattr(trainer, "metrics", {}), "fitness": fitness},
            "date": datetime.now().isoformat(timespec="seconds"),
            "version": ultralytics.__version__,
            "weight_source": "raw",
        }
        temp_ckpt = ckpt.with_suffix(".tmp")
        torch.save(raw_ckpt, temp_ckpt)
        temp_ckpt.replace(ckpt)

        self.records[ckpt.name] = {
            "epoch": int(trainer.epoch),
            "fitness": fitness,
            "weight_source": "raw",
        }

        # Prune: giữ lại đúng keep_top_k checkpoint tốt nhất.
        while len(self.records) > self.keep_top_k:
            worst = min(self.records, key=lambda name: self._quality_key(self.records[name]))
            worst_path = weights_dir / worst
            if worst_path.exists():
                worst_path.unlink()
            del self.records[worst]

        ranking_path.write_text(
            json.dumps(self.records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def delete_checkpoint_artifacts(run_dir):
    """
    Xóa toàn bộ checkpoint tạm trong ``<run_dir>/weights``.

    Chỉ chạy sau khi đã average, evaluation, Excel và export tùy chọn. Các kết
    quả không phải checkpoint (results.csv, plots, args.yaml, Excel) được giữ
    nguyên ở bước này; việc dọn tiếp do tidy_run_dir() lo.
    """
    weights_dir = Path(run_dir) / "weights"
    if not weights_dir.exists():
        return 0

    files = [path for path in weights_dir.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    for path in files:
        path.unlink()

    directories = [path for path in weights_dir.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        directory.rmdir()
    weights_dir.rmdir()

    print(
        f"  ✓ Đã xóa {len(files)} checkpoint/file ranking tạm "
        f"({total_bytes / (1024 ** 2):.1f} MB): {weights_dir}"
    )
    return len(files)


_WORKER_LIMIT_LOGGED = False


def resolve_train_workers():
    """
    Số worker truyền cho model.train(), đã clamp theo CPU job thực sự được cấp.

    Vì sao cần clamp — hai điều Ultralytics làm ngầm:

    1. ``DetectionTrainer.get_dataloader``::

           workers = self.args.workers if mode == "train" else self.args.workers * 2

       nên val loader trong lúc train xin GẤP ĐÔI. Đặt WORKERS=16 thì log hiện
       "32 worker processes" — đó không phải bug của config, mà là ×2 này.

    2. ``build_dataloader`` chỉ chặn bằng ``os.cpu_count()`` (CPU của CẢ NODE),
       trong khi PyTorch cảnh báo dựa trên ``os.sched_getaffinity`` (CPU mà
       SLURM cấp cho job). Trên node nhiều CPU nhưng job xin ít, Ultralytics vẫn
       tạo đủ 2×workers còn PyTorch thì kêu quá tay.

    Cả hai loader sống SONG SONG suốt quá trình train (``_setup_train`` tạo cả
    hai) ⇒ 3 × workers process con. Đặt trần ``workers ≤ cpu_quota // 2`` để
    không loader nào vượt số CPU được cấp — vừa hết warning, vừa bớt RAM.
    Tắt bằng Config.AUTO_LIMIT_WORKERS = False.
    """
    global _WORKER_LIMIT_LOGGED

    requested = int(Config.WORKERS)
    if not Config.AUTO_LIMIT_WORKERS:
        return requested

    quota = cpu_quota()
    allowed = max(1, quota // 2)  # val loader dùng workers×2, phải ≤ quota
    if requested <= allowed:
        return requested

    if not _WORKER_LIMIT_LOGGED:
        _WORKER_LIMIT_LOGGED = True
        print(
            f"  ⚠ WORKERS={requested} → giảm còn {allowed}: job chỉ được cấp {quota} CPU, "
            f"mà Ultralytics dựng val loader với workers×2 ({requested * 2} worker) "
            "nên PyTorch sẽ cảnh báo 'This DataLoader will create ... worker processes'. "
            "Đặt Config.AUTO_LIMIT_WORKERS=False nếu muốn giữ nguyên."
        )
    return allowed


def build_train_args(data_yaml):
    """
    Map Config → kwargs của model.train().
    Chỉ truyền các tham số override, còn lại để mặc định của Ultralytics.
    EXTRA_TRAIN_ARGS được update SAU CÙNG nên có thể ghi đè mọi key chuẩn.
    """
    train_args = {
        # ---- Training core ----
        "data": str(data_yaml),
        "epochs": int(Config.EPOCHS),
        "imgsz": int(Config.IMGSZ),
        "batch": int(Config.BATCH),
        "workers": resolve_train_workers(),
        "seed": int(Config.RANDOM_SEED),
        "patience": int(Config.PATIENCE),
        "pretrained": bool(Config.PRETRAINED),
        "cache": Config.CACHE,
        "resume": bool(Config.RESUME),
        "deterministic": bool(Config.DETERMINISTIC),
        "project": Config.PROJECT,
        "exist_ok": bool(Config.EXIST_OK),

        # ---- Optimizer & LR schedule (Overridden) ----
        "optimizer": Config.OPTIMIZER,
        "lr0": float(Config.LR0),
        "lrf": float(Config.LRF),
        "warmup_epochs": float(Config.WARMUP_EPOCHS),
        "cos_lr": bool(Config.COS_LR),

        # ---- Augmentation (Overridden) ----
        "mixup": float(Config.MIXUP),
        "copy_paste": float(Config.COPY_PASTE),
    }

    if Config.USE_CWA:
        # Lưu ckpt mỗi epoch để có nguồn chọn Top-K; TopKCheckpointManager
        # prune ngay nên disk không phình theo số epoch
        train_args["save_period"] = 1
    else:
        # Không dùng CWA → không cần epoch checkpoint, nhưng VẪN phải
        # save=True: với save=False Ultralytics chỉ gọi save_model() ở epoch
        # cuối nên best.pt có thể không bao giờ được ghi, và Baseline mất
        # weights để đánh giá.
        train_args["save"] = True

    if Config.DEVICE is not None:
        train_args["device"] = Config.DEVICE
    if Config.NAME:
        train_args["name"] = Config.NAME
    train_args.update(Config.EXTRA_TRAIN_ARGS or {})

    # CWA bắt buộc validate và lưu ở mọi epoch để mỗi raw epochN.pt có
    # fitness validation tương ứng. Không cho EXTRA_TRAIN_ARGS phá invariant.
    if Config.USE_CWA:
        train_args["val"] = True
        train_args["save"] = True
        train_args["save_period"] = 1
    return train_args


def _train_one_seed(seed, exp_dir):
    """
    Train + evaluate một seed. Trả về dict kết quả của seed đó.

    Run dir = <exp_dir>/seeds/seed_<seed>; kết thúc chỉ còn Excel + charts/.
    """
    from ultralytics import YOLO

    Config.RANDOM_SEED = seed
    Config.NAME = f"seed_{seed}"
    Config.PROJECT = str(Path(exp_dir) / "seeds")
    # exp_dir đã được đảm bảo rỗng lúc bắt đầu → tên seed_<N> chắc chắn chưa
    # tồn tại; exist_ok=True để Ultralytics không tự đổi thành seed_<N>2.
    Config.EXIST_OK = True

    # Step 1: data với val' độc lập (tách tương ứng theo seed hiện tại)
    data_yaml = prepare_dataset()

    # Step 2: train
    model = YOLO(Config.MODEL)
    if getattr(model, "task", None) != "detect":
        raise ValueError(
            "Pipeline này chỉ hỗ trợ object detection. "
            f"Model {Config.MODEL!r} có task={getattr(model, 'task', None)!r}; "
            "hãy dùng weights/config detection (ví dụ yolov8s.pt)."
        )
    # Swap cls loss NẾU Config.LOSS_FUNCTION != 'bce'.
    # install_cls_loss đăng ký callback vì trainer dựng DetectionModel MỚI;
    # assert_cls_loss_installed kiểm chứng loss thực tế trước khi train chạy.
    install_cls_loss(model)
    model.add_callback("on_train_start", assert_cls_loss_installed)

    if Config.USE_CWA:
        manager = TopKCheckpointManager(Config.KEEP_TOP_K_CHECKPOINTS)
        model.add_callback("on_train_start", manager.on_train_start)
        model.add_callback("on_train_epoch_end", manager.on_train_epoch_end)
        model.add_callback("on_model_save", manager.on_model_save)
        print(
            f"  CWA: giữ Top-{Config.KEEP_TOP_K_CHECKPOINTS} RAW checkpoint "
            "theo raw-model fitness validation cao nhất"
        )

    run_dir = None
    result = None
    try:
        metrics = model.train(**build_train_args(data_yaml))

        run_dir = Path(model.trainer.save_dir)
        print(f"\n  ✓ Training completed cho seed {seed}. Run dir: {run_dir}")

        # Metrics val' dùng để kiểm tra/chọn checkpoint, không phải test report.
        val_metrics = {}
        if metrics is not None:
            from evaluate import extract_overall_metrics

            print_detection_metrics(
                metrics, header=f"FINAL VALIDATION on val' (best ckpt) | Seed {seed}"
            )
            val_metrics = extract_overall_metrics(metrics)
        # DetMetrics giữ tham chiếu ngược tới validator của trainer (và
        # dataloader của nó) — số liệu đã trích xong nên bỏ ngay.
        metrics = None

        # Test chỉ báo cáo hiệu năng cho các K đã định trước; không dùng chọn K.
        evaluation = run_method_evaluation(run_dir, data=data_yaml, seed=seed)

        # Export phải chạy trước cleanup vì best.pt cũng là checkpoint tạm.
        exported_model = (
            export_model(run_dir / "weights" / "best.pt") if Config.EXPORT_ENABLED else None
        )

        result = {
            "seed": seed,
            "run_dir": str(run_dir),
            "status": "ok",
            "checkpoint_policy": "temporary_then_deleted",
            "val_metrics": val_metrics,
            "strategy_records": evaluation["strategy_records"],
            "ranking": evaluation["ranking"],
            "excel": evaluation["excel"],
            "exported_model": exported_model,
        }
        return result
    finally:
        # Kể cả evaluation/export lỗi, không để checkpoint tạm nằm lại.
        if run_dir is None:
            trainer = getattr(model, "trainer", None)
            save_dir = getattr(trainer, "save_dir", None)
            run_dir = Path(save_dir) if save_dir else None

        # Trả RAM về TRƯỚC khi seed kế tiếp dựng dataset/dataloader mới.
        # trainer giữ train_loader + test_loader, mỗi loader giữ WORKERS
        # process con còn sống kèm prefetch queue — không đóng tay thì seed sau
        # bị OOM killer giết ngay lúc build dataset (xem memory.py).
        release_yolo(model)
        del model
        free_memory()

        if run_dir is not None:
            if Config.DELETE_CHECKPOINTS_AFTER_RUN:
                delete_checkpoint_artifacts(run_dir)
            # Chỉ dọn khi seed chạy trọn vẹn — run hỏng giữ nguyên results.csv
            # và log để còn debug được.
            if Config.TIDY_RUN_DIR and result is not None:
                tidy_run_dir(run_dir)


def train_detector():
    """
    Pipeline train hoàn chỉnh:
    1. Tạo experiment group theo --exp-name; fallback timestamp nếu không truyền
    2. Với MỖI seed: tách val' độc lập từ train → train → đánh giá Baseline vs
       CWA (Top-K average) trên split test → Excel + charts của seed đó
    3. Xóa checkpoint tạm, dọn run dir xuống còn Excel + charts
    4. Tổng hợp tất cả seed thành SUMMARY.xlsx (mean ± std) + chart tổng hợp
    """
    # ── Lưu cấu hình gốc ──────────────────────────────────────────────────────
    orig_seeds = Config.RANDOM_SEED
    orig_project = Config.PROJECT      # thư mục root gốc (results/detection)
    orig_exp_name = Config.EXP_NAME
    orig_name = Config.NAME            # None hoặc custom name người dùng đặt
    orig_exist_ok = Config.EXIST_OK

    seeds = ([int(s) for s in orig_seeds]
             if isinstance(orig_seeds, (list, tuple)) else [int(orig_seeds)])

    # Có --exp-name: dùng đúng tên được truyền, dễ quản lý trên server.
    # Không có: fallback timestamp + model để tương thích cách chạy cũ.
    model_stem = Path(str(Config.MODEL)).stem          # "yolov8s" từ "yolov8s.pt"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if orig_exp_name:
        exp_group = str(orig_exp_name).strip()
    else:
        base_name = orig_name if orig_name else "exp"
        exp_group = f"{base_name}_{timestamp}_{model_stem}"
    exp_dir = Path(orig_project) / exp_group

    if orig_exp_name and exp_dir.exists() and any(exp_dir.iterdir()):
        raise FileExistsError(
            f"Experiment đã tồn tại và không rỗng: {exp_dir}\n"
            "Hãy chọn --exp-name khác để kết quả các lần chạy không bị trộn."
        )
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(" CWA - OBJECT DETECTION TRAINING (Ultralytics YOLO)")
    print("=" * 70)
    print(f"  Model      : {Config.MODEL}")
    print(f"  Data       : {Config.DATA} (VOC.yaml built-in sẽ tự download lần đầu)")
    print(f"  Seeds      : {seeds}")
    print(f"  Experiment : {exp_dir}")
    print("=" * 70)

    # ── Snapshot config tại thời điểm chạy ────────────────────────────────────
    config_snapshot = {
        "experiment_group": exp_group,
        "explicit_exp_name": orig_exp_name,
        "timestamp": timestamp,
        "model": Config.MODEL,
        "data": Config.DATA,
        "val_ratio": Config.VAL_RATIO,
        "epochs": Config.EPOCHS,
        "imgsz": Config.IMGSZ,
        "batch": Config.BATCH,
        "seeds": seeds,
        "optimizer": Config.OPTIMIZER,
        "lr0": Config.LR0,
        "lrf": Config.LRF,
        "warmup_epochs": Config.WARMUP_EPOCHS,
        "cos_lr": Config.COS_LR,
        "mixup": Config.MIXUP,
        "copy_paste": Config.COPY_PASTE,
        "patience": Config.PATIENCE,
        "loss_function": Config.LOSS_FUNCTION,
        "focal_gamma": Config.FOCAL_GAMMA,
        "focal_alpha": Config.FOCAL_ALPHA,
        "use_cwa": Config.USE_CWA,
        "top_k_values": Config.TOP_K_VALUES,
        "keep_top_k_checkpoints": Config.KEEP_TOP_K_CHECKPOINTS,
        "averaging": "uniform element-wise mean of learnable parameters (no EMA)",
        "baseline_source": "rank#1 raw FP32" if Config.BASELINE_FROM_RAW_TOPK else "best.pt (FP16)",
        "use_bn_update": Config.USE_BN_UPDATE,
        "bn_update_batches": Config.BN_UPDATE_BATCHES,
        "bn_update_augment": Config.BN_UPDATE_AUGMENT,
        "bn_update_control": Config.BN_UPDATE_CONTROL,
        "eval_split": Config.EVAL_SPLIT,
        "checkpoint_weight_source": "raw" if Config.USE_CWA else "ultralytics_default",
        "delete_checkpoints_after_run": Config.DELETE_CHECKPOINTS_AFTER_RUN,
    }
    config_path = exp_dir / "experiment_config.json"
    config_path.write_text(json.dumps(config_snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✓ Config snapshot saved → {config_path.name}")

    all_runs_results = []
    failures = []

    try:
        for idx, seed in enumerate(seeds):
            # Log RAM đầu mỗi seed: con số này phải ĐI NGANG qua các seed.
            # Nếu nó tăng dần (hoặc "worker process" > 0 lúc bắt đầu) thì còn
            # dataloader/validator bị giữ lại và job sẽ OOM ở seed sau.
            print(
                f"\n>>>> [Seed {idx + 1}/{len(seeds)}] RANDOM_SEED = {seed}"
                f"  |  {memory_report()} <<<<"
            )
            try:
                all_runs_results.append(_train_one_seed(seed, exp_dir))
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                # Một seed hỏng (OOM, lỗi data...) không được làm mất kết quả
                # của các seed đã chạy xong. Ghi nhận rồi chạy tiếp seed sau.
                failures.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})
                print(f"\n  ✗ Seed {seed} THẤT BẠI: {type(exc).__name__}: {exc}")
                traceback.print_exc()
            finally:
                free_memory()
                print(f"  [RAM] Sau seed {seed}: {memory_report()}")
    finally:
        # ── Khôi phục cấu hình gốc kể cả khi có lỗi ───────────────────────────
        Config.RANDOM_SEED = orig_seeds
        Config.PROJECT = orig_project
        Config.EXP_NAME = orig_exp_name
        Config.NAME = orig_name
        Config.EXIST_OK = orig_exist_ok

    if not all_runs_results:
        print("\n  ✗ Không seed nào chạy thành công — không có gì để tổng hợp.")
        if failures:
            for failure in failures:
                print(f"    seed {failure['seed']}: {failure['error']}")
        return all_runs_results

    print_multi_seed_table(all_runs_results)

    summary = export_experiment_summary(
        all_runs_results, exp_dir, config_snapshot=config_snapshot
    )
    if summary:
        _, mean_std_df, per_seed_df, order, shorts, baseline_name = summary
        export_experiment_charts(
            mean_std_df, per_seed_df, order, shorts, baseline_name, exp_dir / "charts"
        )

    print("\n" + "=" * 70)
    print("  ✓ EXPERIMENT COMPLETE")
    print(f"  ✓ Kết quả: {exp_dir}")
    print(f"      SUMMARY.xlsx            mean ± std của {len(all_runs_results)} seed")
    print("      charts/                 chart tổng hợp")
    print("      seeds/seed_<N>/         Excel + charts từng seed")
    if failures:
        print(f"  ⚠ {len(failures)}/{len(seeds)} seed thất bại:")
        for failure in failures:
            print(f"      seed {failure['seed']}: {failure['error']}")
    print("=" * 70)

    return all_runs_results


def export_model(weights):
    """
    Xuất weights đã train sang format deploy (ONNX/TensorRT/...) bằng
    model.export() của Ultralytics — phục vụ hướng Edge AI.
    """
    from ultralytics import YOLO

    export_args = {
        "format": Config.EXPORT_FORMAT,
        "half": bool(Config.EXPORT_HALF),
        "dynamic": bool(Config.EXPORT_DYNAMIC),
        "simplify": bool(Config.EXPORT_SIMPLIFY),
        "imgsz": int(Config.EXPORT_IMGSZ or Config.IMGSZ),
    }
    if Config.EXPORT_DEVICE is not None:
        export_args["device"] = Config.EXPORT_DEVICE

    print(f"\n  [Edge AI] Exporting {weights} → format='{export_args['format']}' ...")
    model = YOLO(str(weights))
    exported_path = model.export(**export_args)
    print(f"  ✓ Exported model: {exported_path}")
    return exported_path
