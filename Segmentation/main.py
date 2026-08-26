"""
Main entrypoint — CWA Instance Segmentation (Ultralytics YOLO-seg).

YOLO segmentation models only. All configuration lives in config.py; the
commonly used values can be overridden on the CLI.

    # Train (model comes from Config.MODEL; give the experiment a clear name)
    python main.py train --exp-name yolov8s_exp1
    #   → results/segmentation/yolov8s_exp1/
    #       README.md, experiment_config.json,
    #       summary/{yolov8s_exp1_summary.xlsx, charts/},
    #       seeds/seed_<N>/{seed_<N>_results.xlsx, charts/, logs/}

    # Re-evaluate Baseline + CWA on an already trained run
    python main.py methods --run-dir results/segmentation/<experiment>/seeds/seed_42

    # Evaluate any weights file (prints mAP/P/R/per-class AP + Excel)
    python main.py eval --weights <run>/weights/best.pt --split test

    # Export Excel from a run dir (offline, needs only results.csv)
    python main.py export --run-dir <run>

    # Edge AI: export weights to ONNX/TensorRT
    python main.py export-model --weights <run>/weights/best.pt

Details on fitness, EMA, averaging and the data split: see README.md
"""
import argparse
import sys

from config import Config

EVAL_SPLIT_CHOICES = ["val", "test", "train"]


def configure_console_encoding():
    """Keep UTF-8 help/log output working on Windows consoles."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                # Some IDEs/notebooks wrap the stream and disallow reconfigure.
                pass


def add_common_args(parser):
    """Arguments shared by every subcommand. Each one overrides Config."""
    parser.add_argument(
        "--model",
        help="Model YOLO-seg: yolov8s-seg.pt, yolo11s-seg.pt, .pt/.yaml segmentation custom... "
             "Override Config.MODEL.",
    )
    parser.add_argument("--data", help="carparts-seg.yaml or a path to a custom segmentation data.yaml.")
    parser.add_argument("--val-ratio", type=float, help="Fraction split off train as val' (0 = no split).")
    parser.add_argument("--epochs", type=int, help="Override Config.EPOCHS.")
    parser.add_argument("--imgsz", type=int, help="Override Config.IMGSZ.")
    parser.add_argument("--batch", type=int, help="Override Config.BATCH (-1 = auto-batch).")
    parser.add_argument("--device", help="Device: 0 | 0,1 | cpu. Auto by default.")
    parser.add_argument("--workers", type=int, help="Override Config.WORKERS.")
    parser.add_argument("--seed", type=int, nargs="+", help="Override Config.RANDOM_SEED (one or more seeds, e.g. --seed 42 100).")
    parser.add_argument("--optimizer", help="Override Config.OPTIMIZER (auto/SGD/AdamW/...).")
    parser.add_argument("--lr0", type=float, help="Override Config.LR0.")
    parser.add_argument("--lrf", type=float, help="Override Config.LRF.")
    parser.add_argument("--patience", type=int, help="Override Config.PATIENCE.")
    parser.add_argument(
        "--loss",
        choices=["bce", "focal"],
        help="Override Config.LOSS_FUNCTION: 'bce' (Ultralytics default) or 'focal' (FocalBCE).",
    )
    parser.add_argument("--focal-gamma", type=float, help="Override Config.FOCAL_GAMMA.")
    parser.add_argument("--focal-alpha", type=float, help="Override Config.FOCAL_ALPHA.")
    parser.add_argument("--project", help="Override Config.PROJECT (root output directory).")
    parser.add_argument(
        "--exp-name",
        "--exp_name",
        dest="exp_name",
        help="Exact experiment directory name; no timestamp is appended.",
    )
    parser.add_argument(
        "--name",
        help="Legacy prefix for the auto-generated name when --exp-name is omitted.",
    )
    parser.add_argument(
        "--split",
        choices=EVAL_SPLIT_CHOICES,
        help="Split used for final reporting (Config.EVAL_SPLIT). Default 'test'.",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="CWA - Instance Segmentation pipeline (Ultralytics YOLO-seg).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser(
        "train",
        help="Train YOLO (val' carved from train), compare Baseline vs CWA "
             "on test, export Excel.",
    )
    add_common_args(p_train)
    p_train.add_argument(
        "--no-cwa", action="store_true",
        help="Disable CWA (no Top-K checkpoint saving or averaging).",
    )
    p_train.add_argument(
        "--export-after-train", action="store_true",
        help="Enable Config.EXPORT_ENABLED: export the model (ONNX/... per config) after training.",
    )

    p_strategies = subparsers.add_parser(
        "methods",
        aliases=["strategies"],
        help="Re-evaluate Baseline + CWA on an already trained run and export Excel.",
    )
    add_common_args(p_strategies)
    p_strategies.add_argument("--run-dir", required=True, help="An Ultralytics run directory that has already been trained.")

    p_eval = subparsers.add_parser(
        "eval", help="Evaluate a weights file with model.val(); print metrics + export Excel."
    )
    add_common_args(p_eval)
    p_eval.add_argument("--weights", help="Trained weights (best.pt). Defaults to Config.MODEL.")
    p_eval.add_argument("--run-dir", help="Run dir containing results.csv, to attach the PerEpoch sheet.")
    p_eval.add_argument("--no-excel", action="store_true", help="Print metrics only; do not export Excel.")

    p_export = subparsers.add_parser(
        "export", help="Export Excel (Summary + PerEpoch) from a trained run dir (offline)."
    )
    add_common_args(p_export)
    p_export.add_argument("--run-dir", required=True, help="Ultralytics run dir (must contain results.csv).")
    p_export.add_argument("--output", dest="excel_output", help="Path file .xlsx output.")

    p_export_model = subparsers.add_parser(
        "export-model", help="Edge AI: export trained weights to ONNX/TensorRT (model.export)."
    )
    add_common_args(p_export_model)
    p_export_model.add_argument("--weights", required=True, help="Trained weights to export.")
    p_export_model.add_argument("--format", dest="export_format", help="Override Config.EXPORT_FORMAT.")
    p_export_model.add_argument("--half", action="store_true", help="Export FP16 (Config.EXPORT_HALF).")

    return parser.parse_args()


def apply_cli_overrides(args):
    """Apply CLI overrides onto Config."""
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

    elif args.command in ("methods", "strategies"):
        Config.validate_config(require_model=False)
        from evaluate import run_method_evaluation

        # data precedence: --data > the run's args.yaml > Config.DATA
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
        # data precedence: --data > run args.yaml (keeps the holdout split) > Config.DATA
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
