"""
Train Ultralytics YOLO cho instance segmentation (Carparts).

Toàn bộ training loop giao cho Ultralytics (model.train): LR schedule,
augmentation, early stopping, best.pt/last.pt, results.csv đều do Ultralytics
quản lý trong run dir. Phần thêm vào cho CWA:

- save_period=1 để Ultralytics lưu checkpoint mỗi epoch, NHƯNG
- TopKCheckpointManager (callback on_model_save) prune NGAY checkpoint ngoài
  Top-K theo FITNESS trên val' → disk chỉ giữ đúng K checkpoint cần thiết
  (+ best.pt/last.pt), không lưu tất cả epoch.

Fitness được lấy trực tiếp từ ``trainer.fitness`` — đúng cùng đại lượng mà
Ultralytics dùng cho best.pt và early stopping. EMA được tắt để validation,
fitness, best.pt và các checkpoint epoch đều dùng raw model weights.

Bố cục thư mục kết quả (xem README.md):

    results/segmentation/<exp_name>/
    ├── README.md                  # bảng mean ± std đọc nhanh
    ├── experiment_config.json
    ├── summary/
    │   ├── <exp_name>_summary.xlsx
    │   └── charts/*.png
    └── seeds/seed_<N>/
        ├── seed_<N>_results.xlsx
        ├── charts/*.png
        └── logs/{results.csv,args.yaml}
"""
import json
import math
import shutil
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from config import Config
from dataset import prepare_dataset
from evaluate import (
    HEADLINE_METRIC,
    RANKING_FILE,
    export_multi_seed_summary,
    extract_overall_metrics,
    print_detection_metrics,
    read_results_csv,
    run_method_evaluation,
)
from losses import install_cls_loss

# Ảnh mẫu Ultralytics dump ra run dir — không phải chart, chiếm hàng chục MB.
SAMPLE_IMAGE_PATTERNS = ("train_batch*.jpg", "val_batch*.jpg", "labels.jpg", "labels_correlogram.jpg")
# File log/text giữ lại nhưng đưa vào logs/ cho gọn.
LOG_FILE_NAMES = ("results.csv", "args.yaml")


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


class TopKCheckpointManager:
    """
    Quản lý raw-model checkpoint cho CWA.

    ``on_train_start`` tắt EMA smoothing. ``on_train_epoch_end`` copy 1:1 raw
    state sang module validation riêng, nhờ đó fitness, early stopping và
    best.pt dùng raw weights mà không chạy inference trên training module.

    ``on_model_save`` lưu raw FP32 model của epoch hiện tại, ghi nhận fitness
    trên val và xóa checkpoint tệ nhất nếu vượt KEEP_TOP_K_CHECKPOINTS.

    Ranking được ghi ra <run_dir>/weights/cwa_checkpoints.json để
    evaluate.rank_checkpoints() dùng lại khi average Top-K (CWA).
    Không đụng tới best.pt / last.pt của Ultralytics.
    """

    def __init__(self, keep_top_k):
        self.keep_top_k = int(keep_top_k)
        self.records = {}  # filename -> {"epoch": int, "fitness": float, "weight_source": "raw"}

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

        # Validator của Ultralytics chạy trong ``torch.inference_mode()`` và
        # một số phiên bản PyTorch/YOLO có thể thay BN running_mean/running_var
        # của validation shadow bằng inference tensors. Ở epoch kế tiếp,
        # ``no_grad()`` vẫn không được phép copy_ vào các tensor đó (PyTorch
        # 2.5 báo "Inplace update to inference tensor outside InferenceMode").
        # Đồng bộ ngay trong inference_mode vừa an toàn cho shadow chỉ dùng
        # validation, vừa giữ training model raw hoàn toàn tách biệt.
        #
        # Lưu ý: BaseValidator ép ``model.half()`` khi val trong lúc train với
        # AMP, nên shadow này có thể là FP16 — load_state_dict sẽ cast. Đây là
        # đúng hành vi mặc định của Ultralytics (EMA cũng được val ở FP16) và
        # KHÔNG ảnh hưởng tới raw FP32 lưu trong epochN.pt.
        with torch.inference_mode():
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
            raise FileNotFoundError(f"Checkpoint vừa lưu không tồn tại: {ckpt}")

        # Ghi đè epoch checkpoint mặc định (field 'ema', FP16) bằng raw FP32
        # model rõ ràng trong field 'model'. Không lưu optimizer vì file này chỉ
        # dùng cho post-training averaging, không dùng resume.
        import torch
        import ultralytics

        # BaseModel.loss() gắn `criterion` (chứa tensor trên GPU) vào model.
        # deepcopy cả criterion làm checkpoint phình ra và buộc lúc load phải
        # dựng lại loss object — tách ra trước khi copy rồi trả lại nguyên trạng.
        source_model = unwrap_ultralytics_model(trainer.model)
        criterion = getattr(source_model, "criterion", None)
        if criterion is not None:
            source_model.criterion = None
        try:
            raw_model = deepcopy(source_model).float().cpu().eval()
        finally:
            if criterion is not None:
                source_model.criterion = criterion
        for parameter in raw_model.parameters():
            parameter.grad = None
            parameter.requires_grad_(False)

        raw_ckpt = {
            "epoch": int(trainer.epoch),
            "best_fitness": float(getattr(trainer, "best_fitness", fitness) or fitness),
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

        # Prune: fitness thấp nhất tệ nhất; nếu hòa thì loại epoch cũ hơn.
        while len(self.records) > self.keep_top_k:
            worst = min(
                self.records,
                key=lambda name: (self.records[name]["fitness"], self.records[name]["epoch"]),
            )
            worst_path = weights_dir / worst
            if worst_path.exists():
                worst_path.unlink()
            del self.records[worst]

        ranking_path.write_text(
            json.dumps(self.records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def cap_val_dataloader_workers(trainer):
    """
    Buộc val dataloader lúc train dùng ĐÚNG ``Config.WORKERS``.

    ``ultralytics/models/yolo/detect/train.py`` có:

        workers = self.args.workers if mode == "train" else self.args.workers * 2

    → set ``workers=16`` thì val loader mở 32 worker, dễ gây
    "DataLoader worker (pid ...) is killed by signal" hoặc hết ``/dev/shm``.
    Callback này chỉ tạm chia đôi ``args.workers`` đúng lúc dựng val loader để
    phép nhân 2 của Ultralytics trả về đúng con số đã cấu hình; train loader
    không bị đụng tới.

    Đăng ký ở ``on_pretrain_routine_start`` vì ``_setup_train`` dựng cả hai
    dataloader ngay sau callback này.
    """
    if not Config.CAP_VAL_WORKERS:
        return

    original_get_dataloader = trainer.get_dataloader

    def get_dataloader(dataset_path, batch_size=16, rank=0, mode="train"):
        if mode == "train":
            return original_get_dataloader(dataset_path, batch_size, rank, mode)
        requested = int(trainer.args.workers)
        trainer.args.workers = requested // 2  # Ultralytics sẽ nhân 2 trở lại
        try:
            return original_get_dataloader(dataset_path, batch_size, rank, mode)
        finally:
            trainer.args.workers = requested

    trainer.get_dataloader = get_dataloader
    print(f"  Dataloader: val workers bị giới hạn về {int(trainer.args.workers)} "
          "(mặc định Ultralytics là workers × 2)")


def delete_checkpoint_artifacts(run_dir):
    """
    Xóa toàn bộ checkpoint tạm trong ``<run_dir>/weights``.

    Hàm này chỉ chạy sau khi đã hoàn tất average, evaluation, Excel và export
    tùy chọn. Các kết quả không phải checkpoint như results.csv, plots, args.yaml
    và Excel được giữ nguyên.
    """
    weights_dir = Path(run_dir) / "weights"
    if not weights_dir.exists():
        return 0

    files = [path for path in weights_dir.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    shutil.rmtree(weights_dir, ignore_errors=True)

    print(
        f"  ✓ Đã xóa {len(files)} checkpoint/file ranking tạm "
        f"({total_bytes / (1024 ** 2):.1f} MB): {weights_dir}"
    )
    return len(files)


def organize_seed_dir(run_dir, seed):
    """
    Dọn run dir của Ultralytics thành bố cục: Excel ở gốc, chart trong charts/,
    log thô trong logs/, và bỏ ảnh mẫu (train_batch*/val_batch*/labels*).

    Chạy sau khi đã export Excel và xóa checkpoint.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return

    charts_dir = run_dir / "charts"
    logs_dir = run_dir / "logs"

    # 1. Ảnh mẫu — không phải chart, chiếm nhiều dung lượng
    removed = 0
    if not Config.KEEP_SAMPLE_IMAGES:
        for pattern in SAMPLE_IMAGE_PATTERNS:
            for path in run_dir.glob(pattern):
                path.unlink()
                removed += 1

    # 2. Chart do Ultralytics sinh (results.png, PR/F1 curve, confusion matrix)
    charts_dir.mkdir(exist_ok=True)
    moved_charts = 0
    for path in list(run_dir.glob("*.png")) + list(run_dir.glob("*.jpg")):
        if path.parent != run_dir:
            continue
        target_name = "training_curves.png" if path.name == "results.png" else path.name
        path.replace(charts_dir / target_name)
        moved_charts += 1

    # 3. Log thô
    logs_dir.mkdir(exist_ok=True)
    for name in LOG_FILE_NAMES:
        path = run_dir / name
        if path.exists():
            path.replace(logs_dir / name)

    # 4. Thư mục rỗng còn sót
    for path in (charts_dir, logs_dir, run_dir / "eval_plots"):
        if path.exists() and not any(path.iterdir()):
            path.rmdir()

    print(
        f"  ✓ Đã dọn thư mục seed {seed}: {moved_charts} chart → charts/, "
        f"log → logs/" + (f", xóa {removed} ảnh mẫu" if removed else "")
    )


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
        "workers": int(Config.WORKERS),
        "seed": int(Config.RANDOM_SEED),
        "patience": int(Config.PATIENCE),
        "pretrained": bool(Config.PRETRAINED),
        "cache": Config.CACHE,
        "resume": bool(Config.RESUME),
        "deterministic": bool(Config.DETERMINISTIC),
        "amp": bool(Config.AMP),
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
        # Không dùng CWA → không cần lưu checkpoint
        train_args["save"] = False

    if Config.DEVICE is not None:
        train_args["device"] = Config.DEVICE
    if Config.NAME:
        train_args["name"] = Config.NAME
    train_args.update(Config.EXTRA_TRAIN_ARGS or {})

    # Invariants của CWA: phải validate và lưu checkpoint ở mọi epoch
    # để mỗi epoch có fitness + epochN.pt tương ứng. Không cho EXTRA_TRAIN_ARGS
    # vô tình phá vỡ hai điều kiện này.
    if Config.USE_CWA:
        train_args["val"] = True
        train_args["save"] = True
        train_args["save_period"] = 1
    return train_args


def print_multi_seed_summary(all_runs_results):
    """In bảng tổng hợp kết quả của nhiều seed chạy độc lập."""
    if not all_runs_results:
        return

    # Union các strategy theo thứ tự xuất hiện — một seed lỗi giữa chừng có thể
    # thiếu strategy, không được để KeyError/IndexError làm hỏng báo cáo.
    strategies = []
    for run in all_runs_results:
        for name in (run.get("strategy_results") or {}):
            if name not in strategies:
                strategies.append(name)
    if not strategies:
        return

    print("\n" + "=" * 80)
    print(" MULTI-SEED RUNS SUMMARY (mAP50 / mAP50-95)")
    print("=" * 80)

    width = max(len(name) for name in strategies) + 2
    header = f"  {'Seed':<6} |" + "".join(f" {name:<{width}} |" for name in strategies)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for run in all_runs_results:
        row = f"  {run['seed']:<6} |"
        results = run.get("strategy_results") or {}
        for name in strategies:
            metrics = results.get(name)
            if metrics is None:
                row += f" {'N/A':<{width}} |"
            else:
                m = extract_overall_metrics(metrics)
                row += f" {m['mAP@0.5']:.4f} / {m[HEADLINE_METRIC]:.4f}".ljust(width + 2) + "|"
        print(row)
    print("=" * 80)


def write_experiment_readme(exp_dir, exp_name, summary, seeds, failed):
    """
    Ghi README.md ở gốc experiment: bảng mean ± std + bản đồ thư mục.
    Mục đích là mở thư mục lên là đọc được kết quả ngay, không cần mở Excel.
    """
    exp_dir = Path(exp_dir)
    mean_std = summary.get("mean_std") if summary else None
    baseline = summary.get("baseline") if summary else None

    lines = [
        f"# {exp_name}",
        "",
        f"- Model: `{Config.MODEL}` | data: `{Config.DATA}`",
        f"- Epochs: {Config.EPOCHS} | imgsz: {Config.IMGSZ} | batch: {Config.BATCH} | "
        f"optimizer: {Config.OPTIMIZER} (lr0={Config.LR0})",
        f"- Seeds: {', '.join(str(s) for s in seeds)}",
        f"- Eval split: `{Config.EVAL_SPLIT}` | Top-K: {Config.TOP_K_VALUES}",
        f"- Averaging: uniform element-wise mean của raw FP32 weights (**không dùng EMA**)",
        f"- BN recalibration: "
        + (f"{Config.BN_UPDATE_BATCHES} batch train, no-grad forward" if Config.USE_BN_UPDATE else "OFF"),
        f"- Ngày export: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]

    if failed:
        lines += [
            "> ⚠ Các seed sau **thất bại** và không có trong bảng tổng hợp: "
            + ", ".join(f"`{seed}` ({err})" for seed, err in failed),
            "",
        ]

    if mean_std is not None and not mean_std.empty:
        lines += [
            f"## Kết quả trên split `{Config.EVAL_SPLIT}` (mean ± std, n={len(seeds) - len(failed)} seeds)",
            "",
            "| Strategy | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |",
            "| --- | --- | --- | --- | --- |",
        ]
        for _, row in mean_std.iterrows():
            def cell(metric):
                key = f"{metric} mean±std"
                return str(row[key]) if key in row else "—"

            lines.append(
                f"| {row['Method']} | {cell('mAP@0.5')} | {cell(HEADLINE_METRIC)} | "
                f"{cell('Precision')} | {cell('Recall')} |"
            )
        lines.append("")
        headline_column = f"{HEADLINE_METRIC} mean"
        if baseline and headline_column in mean_std.columns:
            best = mean_std.loc[mean_std[headline_column].idxmax(), "Method"]
            lines += [f"**Tốt nhất theo {HEADLINE_METRIC} (mean): `{best}`** — baseline là `{baseline}`.", ""]

    lines += [
        "## Bố cục thư mục",
        "",
        "```text",
        f"{exp_name}/",
        "├── README.md                     # file này",
        "├── experiment_config.json        # snapshot config lúc chạy",
        "├── summary/",
        f"│   ├── {exp_name}_summary.xlsx   # MeanStd | DeltaVsBaseline | PerSeed | PerClass | Checkpoints",
        "│   └── charts/                   # chart tổng hợp (mean±std, đường cong theo K, ...)",
        "└── seeds/",
        "    └── seed_<N>/",
        "        ├── seed_<N>_results.xlsx # Summary | PerEpoch | Checkpoints của riêng seed",
        "        ├── charts/               # training curves, PR/F1, confusion matrix, checkpoint selection",
        "        └── logs/                 # results.csv, args.yaml",
        "```",
        "",
        "Checkpoint `.pt` **không** được lưu lại: chúng chỉ tồn tại tạm trong lúc",
        "rank / average / BN update / evaluate rồi bị xóa "
        "(`Config.DELETE_CHECKPOINTS_AFTER_RUN = True`).",
        "",
    ]

    readme_path = exp_dir / "README.md"
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    return readme_path


def train_detector():
    """
    Pipeline train hoàn chỉnh cho instance segmentation:
    1. Tạo experiment group theo --exp-name; fallback timestamp nếu không truyền
    2. Chuẩn bị data (tách val' nếu cần theo VAL_RATIO, hỗ trợ seed dạng list/int)
    3. Loop qua tất cả các seed → mỗi seed = seeds/seed_<N> bên trong experiment
    4. Huấn luyện model.train() với TopKCheckpointManager (nếu USE_CWA)
    5. Đánh giá Baseline / Top-1 raw / CWA (Top-K average) trên test
    6. Xuất Excel + chart, Edge AI export nếu được bật
    7. Xóa checkpoint tạm, dọn thư mục seed, ghi summary + README cho experiment
    """
    # Import trễ để validate config / --help không cần ultralytics
    from ultralytics import YOLO

    # ── Lưu cấu hình gốc ──────────────────────────────────────────────────────
    orig_seeds = Config.RANDOM_SEED
    orig_project = Config.PROJECT          # thư mục root gốc (results/segmentation)
    orig_exp_name = Config.EXP_NAME
    orig_name = Config.NAME                # None hoặc custom name người dùng đặt

    # Chuẩn hóa seeds thành list (loại trùng, giữ thứ tự)
    if isinstance(orig_seeds, (list, tuple)):
        seeds = list(dict.fromkeys(int(s) for s in orig_seeds))
    else:
        seeds = [int(orig_seeds)]

    # ── Tạo thư mục experiment group ─────────────────────────────────────────
    # Có --exp-name: dùng ĐÚNG tên người dùng đặt, dễ quản lý trên server.
    # Không có: fallback tên tự động timestamp + model để tương thích code cũ.
    model_stem = Path(str(Config.MODEL)).stem          # "yolov8s-seg" từ "yolov8s-seg.pt"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if orig_exp_name:
        exp_group = str(orig_exp_name).strip()
    else:
        base_name = orig_name if orig_name else "exp"
        exp_group = f"{base_name}_{timestamp}_{model_stem}"
    exp_dir = Path(orig_project) / exp_group
    seeds_dir = exp_dir / "seeds"
    summary_dir = exp_dir / "summary"

    # Tên explicit phải trỏ tới một experiment mới để không trộn nhiều lần chạy.
    if orig_exp_name and exp_dir.exists() and any(exp_dir.iterdir()):
        raise FileExistsError(
            f"Experiment đã tồn tại và không rỗng: {exp_dir}\n"
            "Hãy chọn --exp-name khác để kết quả các lần chạy không bị trộn."
        )
    seeds_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Redirect PROJECT → <exp>/seeds; mỗi seed tạo subfolder seed_<N> trong đây
    Config.PROJECT = str(seeds_dir)

    print("\n" + "=" * 70)
    print(" CWA - INSTANCE SEGMENTATION TRAINING (Ultralytics YOLO-seg)")
    print("=" * 70)
    print(f"  Model      : {Config.MODEL}")
    print(f"  Data       : {Config.DATA}")
    print(f"  Seeds      : {seeds}")
    print(f"  Experiment : {exp_dir}")
    print("=" * 70)

    # ── Lưu snapshot config tại thời điểm chạy ────────────────────────────────
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
        "workers": Config.WORKERS,
        "cap_val_workers": Config.CAP_VAL_WORKERS,
        "amp": Config.AMP,
        "loss_function": Config.LOSS_FUNCTION,
        "focal_gamma": Config.FOCAL_GAMMA,
        "focal_alpha": Config.FOCAL_ALPHA,
        "use_cwa": Config.USE_CWA,
        "top_k_values": Config.TOP_K_VALUES,
        "keep_top_k_checkpoints": Config.KEEP_TOP_K_CHECKPOINTS,
        "eval_top1_baseline": Config.EVAL_TOP1_BASELINE,
        "use_bn_update": Config.USE_BN_UPDATE,
        "bn_update_batches": Config.BN_UPDATE_BATCHES,
        "bn_update_close_mosaic": Config.BN_UPDATE_CLOSE_MOSAIC,
        "eval_split": Config.EVAL_SPLIT,
        "delete_checkpoints_after_run": Config.DELETE_CHECKPOINTS_AFTER_RUN,
    }
    config_path = exp_dir / "experiment_config.json"
    config_path.write_text(json.dumps(config_snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✓ Config snapshot saved → {config_path.name}")

    all_runs_results = []
    failed_seeds = []

    try:
        for idx, seed in enumerate(seeds):
            print(f"\n>>>> [Seed {idx + 1}/{len(seeds)}] Bắt đầu train với RANDOM_SEED = {seed} <<<<")

            # Ghi đè seed, đặt tên subfolder = seed_<N>
            Config.RANDOM_SEED = seed
            Config.NAME = f"seed_{seed}"

            run_dir = None
            model = None
            try:
                # Step 1: data với val' độc lập (tách tương ứng theo seed hiện tại)
                data_yaml = prepare_dataset()

                # Step 2: train
                model = YOLO(Config.MODEL)
                if getattr(model, "task", None) != "segment":
                    raise ValueError(
                        "Pipeline này chỉ hỗ trợ instance segmentation. "
                        f"Model {Config.MODEL!r} có task={getattr(model, 'task', None)!r}; "
                        "hãy dùng weights/config segmentation (ví dụ yolov8s-seg.pt)."
                    )
                # Swap cls loss NẾU Config.LOSS_FUNCTION != 'bce'
                # ('bce' = giữ nguyên v8SegmentationLoss mặc định của Ultralytics)
                install_cls_loss(model)
                model.add_callback("on_pretrain_routine_start", cap_val_dataloader_workers)

                if Config.USE_CWA:
                    manager = TopKCheckpointManager(Config.KEEP_TOP_K_CHECKPOINTS)
                    model.add_callback("on_train_start", manager.on_train_start)
                    model.add_callback("on_train_epoch_end", manager.on_train_epoch_end)
                    model.add_callback("on_model_save", manager.on_model_save)
                    print(
                        f"  CWA: giữ Top-{Config.KEEP_TOP_K_CHECKPOINTS} RAW checkpoint "
                        "theo raw-model fitness cao nhất"
                    )

                # model.train() trả về SegmentMetrics của lượt val CUỐI trên best.pt.
                metrics = model.train(**build_train_args(data_yaml))

                run_dir = Path(model.trainer.save_dir)
                best_weights = run_dir / "weights" / "best.pt"
                print(f"\n  ✓ Training completed for seed {seed}. Run dir: {run_dir}")

                # Step 3: metrics trên val' (holdout) — KHÔNG phải số liệu báo cáo cuối
                val_metrics = {}
                if metrics is not None:
                    print_detection_metrics(
                        metrics,
                        header=f"FINAL VALIDATION on val' (best.pt) | Seed {seed}",
                    )
                    val_metrics = extract_overall_metrics(metrics)

                # Step 4: báo cáo cuối trên EVAL_SPLIT (mặc định test độc lập).
                evaluation = run_method_evaluation(run_dir, data=data_yaml, seed=seed)

                # Step 5: chart minh họa tiêu chí chọn checkpoint (cần results.csv,
                # không cần file .pt) — vẽ trước khi dọn cho chắc chắn.
                _plot_checkpoint_selection(run_dir, evaluation["checkpoints"], seed)

                # Step 6: Edge AI hook (tắt mặc định). Export phải chạy trước cleanup.
                exported_model = export_model(best_weights) if Config.EXPORT_ENABLED else None

                all_runs_results.append({
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "checkpoint_policy": "temporary_then_deleted",
                    "val_metrics": val_metrics,
                    "strategy_results": evaluation["strategies"],
                    "checkpoints": evaluation["checkpoints"],
                    "excel": str(evaluation["excel"]) if evaluation["excel"] else None,
                    "exported_model": exported_model,
                })
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                # Một seed lỗi không được xóa sổ 4 seed đã chạy xong trước đó.
                failed_seeds.append((seed, f"{type(exc).__name__}: {exc}"))
                print(f"\n  ✗ SEED {seed} THẤT BẠI: {type(exc).__name__}: {exc}")
                traceback.print_exc()
            finally:
                # Kể cả evaluation/export lỗi, không để checkpoint tạm nằm lại.
                cleanup_dir = run_dir
                if cleanup_dir is None and model is not None:
                    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
                    cleanup_dir = Path(save_dir) if save_dir else None
                if cleanup_dir is not None:
                    if Config.DELETE_CHECKPOINTS_AFTER_RUN:
                        delete_checkpoint_artifacts(cleanup_dir)
                    organize_seed_dir(cleanup_dir, seed)
    finally:
        # ── Khôi phục cấu hình gốc kể cả khi bị Ctrl-C ────────────────────────
        Config.RANDOM_SEED = orig_seeds
        Config.PROJECT = orig_project
        Config.EXP_NAME = orig_exp_name
        Config.NAME = orig_name

    if len(all_runs_results) > 1:
        print_multi_seed_summary(all_runs_results)

    # Summary + chart cho toàn experiment
    summary = export_multi_seed_summary(
        all_runs_results, summary_dir, exp_name=exp_group,
        extra_info=[("failed_seeds", ", ".join(str(s) for s, _ in failed_seeds) or "none")],
    )
    readme = write_experiment_readme(exp_dir, exp_group, summary, seeds, failed_seeds)

    print("\n" + "=" * 70)
    print("  ✓ EXPERIMENT COMPLETE")
    print(f"  ✓ Results       : {exp_dir}")
    print(f"  ✓ Đọc nhanh     : {readme}")
    if summary.get("excel"):
        print(f"  ✓ Excel tổng hợp: {summary['excel']}")
    if failed_seeds:
        print(f"  ⚠ Seed thất bại : {[s for s, _ in failed_seeds]}")
    print("=" * 70)

    return all_runs_results


def _plot_checkpoint_selection(run_dir, checkpoints, seed):
    """Vẽ chart fitness/epoch + đánh dấu Top-K đã chọn cho một seed."""
    if not Config.MAKE_CHARTS or not checkpoints:
        return
    try:
        import charts as charts_module

        per_epoch_df = read_results_csv(run_dir)
        ranked = [
            (Path(row["Checkpoint"]), float(row["Fitness (val)"]), int(row["Epoch (0-based)"]))
            for row in checkpoints
        ]
        charts_module.plot_checkpoint_selection(
            per_epoch_df, ranked, Path(run_dir) / "charts" / "checkpoint_selection.png", seed=seed
        )
    except Exception as exc:
        print(f"  ⚠ Bỏ qua chart checkpoint selection (seed {seed}): {exc}")


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
