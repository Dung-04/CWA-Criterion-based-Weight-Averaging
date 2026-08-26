"""
Main entrypoint — CWA Instance Segmentation (Ultralytics YOLO-seg).

Nhánh này CHỈ dùng model YOLO. Mọi config chỉnh trong config.py; các giá trị
hay dùng override được qua CLI (pattern như repo gốc).

    # Train (model lấy từ Config.MODEL, đặt tên experiment rõ ràng trên server)
    python main.py train --exp-name yolov8s_exp1
    #   → results/segmentation/yolov8s_exp1/
    #       README.md, experiment_config.json,
    #       summary/{yolov8s_exp1_summary.xlsx, charts/},
    #       seeds/seed_<N>/{seed_<N>_results.xlsx, charts/, logs/}

    # Đánh giá lại Baseline + CWA trên một run đã train
    python main.py strategies --run-dir results/segmentation/<experiment>/seeds/seed_42

    # Eval một file weights bất kỳ (in mAP/P/R/per-class AP + Excel)
    python main.py eval --weights <run>/weights/best.pt --split test

    # Export Excel từ run dir (offline, chỉ cần results.csv)
    python main.py export --run-dir <run>

    # Edge AI: xuất weights sang ONNX/TensorRT
    python main.py export-model --weights <run>/weights/best.pt

Chi tiết về fitness, EMA, averaging và data split: README.md
"""
import argparse
import sys

from config import Config

EVAL_SPLIT_CHOICES = ["val", "test", "train"]


def configure_console_encoding():
    """Cho phép help/log tiếng Việt chạy ổn định trên Windows console."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                # Một số IDE/notebook bọc stream và không cho reconfigure.
                pass


def add_common_args(parser):
    """Args chung cho mọi subcommand — mọi giá trị đều override Config."""
    parser.add_argument(
        "--model",
        help="Model YOLO-seg: yolov8s-seg.pt, yolo11s-seg.pt, .pt/.yaml segmentation custom... "
             "Override Config.MODEL.",
    )
    parser.add_argument("--data", help="carparts-seg.yaml hoặc path data.yaml segmentation custom.")
    parser.add_argument("--val-ratio", type=float, help="Tỉ lệ tách val' từ train (0 = không tách).")
    parser.add_argument("--epochs", type=int, help="Override Config.EPOCHS.")
    parser.add_argument("--imgsz", type=int, help="Override Config.IMGSZ.")
    parser.add_argument("--batch", type=int, help="Override Config.BATCH (-1 = auto-batch).")
    parser.add_argument("--device", help="Device: 0 | 0,1 | cpu. Mặc định auto.")
    parser.add_argument("--workers", type=int, help="Override Config.WORKERS.")
    parser.add_argument("--seed", type=int, nargs="+", help="Override Config.RANDOM_SEED (chấp nhận 1 hoặc nhiều seed, ví dụ: --seed 42 100).")
    parser.add_argument("--optimizer", help="Override Config.OPTIMIZER (auto/SGD/AdamW/...).")
    parser.add_argument("--lr0", type=float, help="Override Config.LR0.")
    parser.add_argument("--lrf", type=float, help="Override Config.LRF.")
    parser.add_argument("--patience", type=int, help="Override Config.PATIENCE.")
    parser.add_argument(
        "--loss",
        choices=["bce", "focal"],
        help="Override Config.LOSS_FUNCTION: 'bce' (mặc định Ultralytics) hoặc 'focal' (FocalBCE).",
    )
    parser.add_argument("--focal-gamma", type=float, help="Override Config.FOCAL_GAMMA.")
    parser.add_argument("--focal-alpha", type=float, help="Override Config.FOCAL_ALPHA.")
    parser.add_argument("--project", help="Override Config.PROJECT (thư mục output gốc).")
    parser.add_argument(
        "--exp-name",
        "--exp_name",
        dest="exp_name",
        help="Tên chính xác của thư mục experiment trên server, không tự ghép timestamp.",
    )
    parser.add_argument(
        "--name",
        help="Prefix legacy cho tên tự động khi không truyền --exp-name.",
    )
    parser.add_argument(
        "--split",
        choices=EVAL_SPLIT_CHOICES,
        help="Split cho báo cáo cuối (Config.EVAL_SPLIT). Mặc định 'test'.",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="CWA - Instance Segmentation pipeline (Ultralytics YOLO-seg).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser(
        "train",
        help="Train YOLO (val' tách riêng từ train) → so sánh Baseline vs "
             "CWA trên test → export Excel.",
    )
    add_common_args(p_train)
    p_train.add_argument(
        "--no-cwa", action="store_true",
        help="Tắt CWA (không lưu/average Top-K checkpoint).",
    )
    p_train.add_argument(
        "--export-after-train", action="store_true",
        help="Bật Config.EXPORT_ENABLED: xuất model (ONNX/... theo config) sau train.",
    )

    p_strategies = subparsers.add_parser(
        "strategies",
        help="Đánh giá lại Baseline + CWA trên một run đã train + export Excel.",
    )
    add_common_args(p_strategies)
    p_strategies.add_argument("--run-dir", required=True, help="Run dir Ultralytics đã train.")

    p_eval = subparsers.add_parser(
        "eval", help="Eval một file weights bằng model.val(), in metrics + export Excel."
    )
    add_common_args(p_eval)
    p_eval.add_argument("--weights", help="Weights đã train (best.pt). Mặc định dùng Config.MODEL.")
    p_eval.add_argument("--run-dir", help="Run dir chứa results.csv để ghép sheet PerEpoch.")
    p_eval.add_argument("--no-excel", action="store_true", help="Chỉ in metrics, không export Excel.")

    p_export = subparsers.add_parser(
        "export", help="Export Excel (Summary + PerEpoch) từ run dir đã train (offline)."
    )
    add_common_args(p_export)
    p_export.add_argument("--run-dir", required=True, help="Run dir Ultralytics (chứa results.csv).")
    p_export.add_argument("--output", dest="excel_output", help="Path file .xlsx output.")

    p_export_model = subparsers.add_parser(
        "export-model", help="Edge AI: xuất weights đã train sang ONNX/TensorRT (model.export)."
    )
    add_common_args(p_export_model)
    p_export_model.add_argument("--weights", required=True, help="Weights đã train cần export.")
    p_export_model.add_argument("--format", dest="export_format", help="Override Config.EXPORT_FORMAT.")
    p_export_model.add_argument("--half", action="store_true", help="Export FP16 (Config.EXPORT_HALF).")

    return parser.parse_args()


def apply_cli_overrides(args):
    """CLI override lên Config — pattern giữ nguyên từ repo gốc."""
    overrides = [
        ("MODEL", getattr(args, "model", None)),
        ("DATA", getattr(args, "data", None)),
        ("VAL_RATIO", getattr(args, "val_ratio", None)),
        ("EPOCHS", getattr(args, "epochs", None)),
        ("IMGSZ", getattr(args, "imgsz", None)),
        ("BATCH", getattr(args, "batch", None)),
        ("DEVICE", getattr(args, "device", None)),
        ("WORKERS", getattr(args, "workers", None)),
        ("RANDOM_SEED", getattr(args, "seed", None)),
        ("OPTIMIZER", getattr(args, "optimizer", None)),
        ("LR0", getattr(args, "lr0", None)),
        ("LRF", getattr(args, "lrf", None)),
        ("PATIENCE", getattr(args, "patience", None)),
        ("LOSS_FUNCTION", getattr(args, "loss", None)),
        ("FOCAL_GAMMA", getattr(args, "focal_gamma", None)),
        ("FOCAL_ALPHA", getattr(args, "focal_alpha", None)),
        ("PROJECT", getattr(args, "project", None)),
        ("EXP_NAME", getattr(args, "exp_name", None)),
        ("NAME", getattr(args, "name", None)),
        ("EVAL_SPLIT", getattr(args, "split", None)),
        ("EXCEL_OUTPUT", getattr(args, "excel_output", None)),
        ("EXPORT_FORMAT", getattr(args, "export_format", None)),
    ]
    for attr, value in overrides:
        if value is not None:
            setattr(Config, attr, value)

    if getattr(args, "no_cwa", False):
        Config.USE_CWA = False
    if getattr(args, "export_after_train", False):
        Config.EXPORT_ENABLED = True
    if getattr(args, "half", False):
        Config.EXPORT_HALF = True


def main():
    args = parse_args()
    apply_cli_overrides(args)

    if args.command == "train":
        Config.validate_config(require_model=True)
        from train import train_detector

        train_detector()

    elif args.command == "strategies":
        Config.validate_config(require_model=False)
        from evaluate import run_method_evaluation

        # data ưu tiên: --data > args.yaml của run > Config.DATA
        data = args.data or None
        run_method_evaluation(args.run_dir, data=data)

    elif args.command == "eval":
        Config.validate_config(require_model=False)
        from evaluate import (
            evaluate_weights,
            export_to_excel,
            infer_run_dir,
            resolve_data_from_run,
        )

        weights = args.weights or Config.MODEL
        run_dir = args.run_dir or infer_run_dir(weights)
        # data ưu tiên: --data > args.yaml của run (giữ đúng holdout split) > Config.DATA
        data = args.data or (resolve_data_from_run(run_dir) if run_dir else None) or Config.DATA
        metrics = evaluate_weights(weights, data, split=Config.EVAL_SPLIT)
        if not args.no_excel:
            target_dir = run_dir if run_dir else getattr(metrics, "save_dir", ".")
            export_to_excel(
                target_dir,
                strategy_results={"Eval": metrics},
                data=data,
                split=Config.EVAL_SPLIT,
            )

    elif args.command == "export":
        Config.validate_config(require_model=False)
        from evaluate import export_to_excel

        export_to_excel(args.run_dir, strategy_results=None)

    elif args.command == "export-model":
        Config.validate_config(require_model=False)
        from train import export_model

        export_model(args.weights)


if __name__ == "__main__":
    configure_console_encoding()
    main()
