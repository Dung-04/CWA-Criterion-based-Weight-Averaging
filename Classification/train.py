"""Train and evaluate classification models with Baseline and CWA.

One entrypoint for every dataset. ``--dataset`` selects the config, which in
turn resolves the loader, split, number of classes, image size, normalization,
augmentation policy and evaluation settings.

    python train.py --dataset cifar100
    python train.py --dataset burmese --model vit_base
    python train.py --dataset tinyimagenet --model vit_base --seed 42

Without --seed, every seed in Config.SEEDS runs sequentially, each in its own
process and output folder.
"""
import argparse
import os
import shutil
import subprocess
import sys


def configure_console_encoding():
    """Keep UTF-8 log output working on Windows consoles (cp1252 by default)."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def _preparse_environment_args():
    """Apply CUDA-related CLI args before torch is imported."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", "--cuda-visible-devices", dest="cuda_visible_devices")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cublas-workspace-config", default=":4096:8")
    args, _ = parser.parse_known_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", args.cublas_workspace_config)


configure_console_encoding()
_preparse_environment_args()

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from configs import Config, DATASET_CHOICES, select_dataset
from data import concatenate_datasets, create_dataloaders, load_dataset
from evaluation import (
    create_performance_charts,
    export_results_to_excel,
    results_to_frames,
    save_confusion_matrices,
)
from methods import BASELINE_NAME, evaluate_all_methods
from models import SUPPORTED_MODELS, parse_model_list
from training import train_model
from utils import (
    export_run_config,
    get_next_run_folder,
    print_dataset_statistics,
    sanitize_run_name,
    set_seed,
)

METRIC_KEYS = ['Test Loss', 'Accuracy (%)', 'Precision (%)', 'Recall (%)',
               'F1-Score (%)', 'AUC (%)']


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate classification models with Baseline and CWA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASET_CHOICES,
        help="Dataset to run. Resolves loader, split, classes and augmentation.",
    )
    parser.add_argument(
        "--model", "--models",
        dest="models",
        nargs="+",
        help=f"Model(s) to run: space/comma separated, or 'all'. "
             f"Supported: {', '.join(SUPPORTED_MODELS)}",
    )
    parser.add_argument(
        "--data-root", "--dataset-path",
        dest="data_root",
        help="Override the dataset location (DATA_ROOT or DATASET_PATH).",
    )
    parser.add_argument("--results-dir", "--output-dir", dest="results_dir",
                        help="Base directory for run outputs.")
    parser.add_argument("--checkpoints-dir",
                        help="Optional base directory for training checkpoints. "
                             "A per-run subfolder is still created.")
    parser.add_argument("--run-name", help="Optional readable name for this run folder.")
    parser.add_argument("--gpu", "--cuda-visible-devices", dest="cuda_visible_devices",
                        help="CUDA_VISIBLE_DEVICES value, e.g. 0, 1, or 0,1.")

    parser.add_argument("--batch-size", type=int, help="Override Config.BATCH_SIZE.")
    parser.add_argument("--epochs", type=int, help="Override Config.NUM_EPOCHS.")
    parser.add_argument("--warmup-epochs", type=int, help="Override Config.WARMUP_EPOCHS.")
    parser.add_argument("--eta-min", type=float, help="Override Config.ETA_MIN.")
    parser.add_argument("--early-stopping", type=int,
                        help="Override Config.EARLY_STOPPING_PATIENCE.")
    parser.add_argument("--fc-layers", nargs="+", type=int,
                        help="Hidden layer sizes for the classifier head, "
                             "e.g. --fc-layers 256 128.")
    parser.add_argument("--dropout", type=float, help="Override Config.DROPOUT_RATE.")
    parser.add_argument("--num-workers", type=int, help="Override Config.NUM_WORKERS.")
    parser.add_argument("--lr", type=float, help="Override Config.LEARNING_RATE.")
    parser.add_argument("--weight-decay", type=float, help="Override Config.WEIGHT_DECAY.")
    parser.add_argument("--optimizer", help="Override Config.OPTIMIZER "
                                            "(adamw/adam/sgd/rmsprop).")
    parser.add_argument("--momentum", type=float,
                        help="Override Config.SGD_MOMENTUM (used by sgd and rmsprop).")
    parser.add_argument("--no-nesterov", action="store_true",
                        help="Disable Nesterov momentum for --optimizer sgd.")
    parser.add_argument("--no-fused-optimizer", action="store_true",
                        help="Disable fused optimizer kernels even on CUDA.")

    parser.add_argument("--seed", type=int, help="Run a single seed.")
    parser.add_argument("--seeds", nargs="+", type=int,
                        help="Run these seeds sequentially. Defaults to Config.SEEDS.")

    parser.add_argument("--top-k", nargs="+", type=int, dest="top_k",
                        help="K values reported for CWA. Override Config.TOP_K_VALUES.")
    parser.add_argument("--keep-top-k", type=int,
                        help="Checkpoints retained for averaging. Must be >= max(--top-k). "
                             "Override Config.KEEP_TOP_K_CHECKPOINTS.")
    parser.add_argument("--save-method-checkpoints", action="store_true",
                        help="Keep the Baseline/CWA weight files after evaluation.")

    parser.add_argument("--cv", action="store_true", help="Enable cross-validation.")
    parser.add_argument("--no-cv", action="store_true", help="Disable cross-validation.")
    parser.add_argument("--cv-splits", type=int, help="Override Config.CV_N_SPLITS.")

    parser.add_argument("--weighted-sampler", action="store_true",
                        help="Enable WeightedRandomSampler.")
    parser.add_argument("--no-weighted-sampler", action="store_true",
                        help="Disable WeightedRandomSampler.")
    parser.add_argument("--auto-delete-checkpoints", action="store_true",
                        help="Delete training checkpoints after evaluation.")
    parser.add_argument("--keep-checkpoints", action="store_true",
                        help="Keep training checkpoints after evaluation.")

    parser.add_argument("--profile-batches", type=int,
                        help="Print DataLoader and compute timing for the first N batches.")
    parser.add_argument("--dataset-stats", action="store_true",
                        help="Print image-size statistics before training "
                             "(opens up to 1000 images).")
    parser.add_argument("--check-dataset", action="store_true",
                        help="Verify every image decodes, then exit.")
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic CUDA behavior. Slower, more reproducible.")
    parser.add_argument("--cublas-workspace-config", default=":4096:8",
                        help="CUBLAS_WORKSPACE_CONFIG used only with --deterministic.")
    return parser.parse_args()


def apply_cli_overrides(args):
    """Apply CLI overrides onto the selected dataset config."""
    models = parse_model_list(args.models)
    if models:
        Config.MODELS = models

    if args.data_root is not None:
        # Datasets name their root differently; set whichever the config uses.
        if hasattr(Config, "DATASET_PATH") and Config.LOADER == "image_folder":
            Config.DATASET_PATH = args.data_root
        else:
            Config.DATA_ROOT = args.data_root

    overrides = [
        ("RESULTS_DIR", args.results_dir),
        ("CHECKPOINTS_DIR", args.checkpoints_dir),
        ("BATCH_SIZE", args.batch_size),
        ("NUM_EPOCHS", args.epochs),
        ("WARMUP_EPOCHS", args.warmup_epochs),
        ("ETA_MIN", args.eta_min),
        ("EARLY_STOPPING_PATIENCE", args.early_stopping),
        ("DROPOUT_RATE", args.dropout),
        ("NUM_WORKERS", args.num_workers),
        ("PROFILE_BATCHES", args.profile_batches),
        ("RANDOM_SEED", args.seed),
        ("LEARNING_RATE", args.lr),
        ("WEIGHT_DECAY", args.weight_decay),
        ("OPTIMIZER", args.optimizer),
        ("SGD_MOMENTUM", args.momentum),
        ("CV_N_SPLITS", args.cv_splits),
        ("TOP_K_VALUES", args.top_k),
        ("KEEP_TOP_K_CHECKPOINTS", args.keep_top_k),
    ]
    for attr, value in overrides:
        if value is not None:
            setattr(Config, attr, value)

    if args.fc_layers is not None:
        Config.CLASSIFIER_CONFIG = args.fc_layers

    if args.no_nesterov:
        Config.SGD_NESTEROV = False

    if args.no_fused_optimizer:
        Config.USE_FUSED_OPTIMIZER = False

    if args.dataset_stats:
        Config.PRINT_DATASET_STATS = True

    if args.save_method_checkpoints:
        Config.SAVE_METHOD_CHECKPOINTS = True

    if args.epochs is not None and args.warmup_epochs is None:
        Config.WARMUP_EPOCHS = min(5, max(1, int(Config.NUM_EPOCHS * 0.1)))

    if args.cv and args.no_cv:
        raise ValueError("Use only one of --cv or --no-cv.")
    if args.cv:
        Config.USE_CROSS_VALIDATION = True
    if args.no_cv:
        Config.USE_CROSS_VALIDATION = False

    if args.weighted_sampler and args.no_weighted_sampler:
        raise ValueError("Use only one of --weighted-sampler or --no-weighted-sampler.")
    if args.weighted_sampler:
        Config.USE_WEIGHTED_SAMPLER = True
    if args.no_weighted_sampler:
        Config.USE_WEIGHTED_SAMPLER = False

    if args.auto_delete_checkpoints and args.keep_checkpoints:
        raise ValueError("Use only one of --auto-delete-checkpoints or --keep-checkpoints.")
    if args.auto_delete_checkpoints:
        Config.AUTO_DELETE_CHECKPOINTS = True
    if args.keep_checkpoints:
        Config.AUTO_DELETE_CHECKPOINTS = False


def run_seed_jobs_if_needed(args):
    """Run each configured seed as a separate process with its own output folder."""
    if args.seed is not None:
        return False

    seeds = args.seeds if args.seeds else getattr(Config, "SEEDS", None)
    if not seeds:
        return False

    seeds = [int(seed) for seed in seeds]
    if len(seeds) <= 1:
        Config.RANDOM_SEED = seeds[0]
        return False

    base_args = sys.argv[1:]
    original_run_name = args.run_name

    print("\n" + "=" * 70)
    print(f" MULTI-SEED RUN: {seeds}")
    print("=" * 70)

    for seed in seeds:
        child_args = base_args + ["--seed", str(seed)]
        if original_run_name:
            child_args += ["--run-name", f"{sanitize_run_name(original_run_name)}_seed{seed}"]
        else:
            model_part = "_".join(Config.MODELS) if Config.MODELS else "models"
            child_args += ["--run-name",
                           f"{args.dataset}_{sanitize_run_name(model_part)}_seed{seed}"]

        print(f"\n[Seed {seed}] Starting: {sys.executable} "
              f"{os.path.basename(__file__)} {' '.join(child_args)}")
        subprocess.run([sys.executable, __file__, *child_args], check=True)

    print("\n" + "=" * 70)
    print(" MULTI-SEED RUN COMPLETED")
    print("=" * 70)
    return True


def save_model_results(model_name, results, output_dir):
    """Save one model's results to Excel (macro + per-class sheets)."""
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    df_macro, df_per_class = results_to_frames({model_name: results})

    excel_path = os.path.join(model_dir, f'{model_name}_results.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_macro.to_excel(writer, sheet_name='Macro Results', index=False)
        df_per_class.to_excel(writer, sheet_name='Per-Class Results', index=False)
    print(f"  ✓ Results saved to: {excel_path}")

    return df_macro


def delete_model_checkpoints(model_name, checkpoints_dir):
    """Delete all training checkpoints for one model."""
    model_checkpoint_dir = os.path.join(checkpoints_dir, model_name)

    if not os.path.exists(model_checkpoint_dir):
        print(f"  ⚠ No checkpoints found at: {model_checkpoint_dir}")
        return

    try:
        shutil.rmtree(model_checkpoint_dir)
        print(f"  ✓ Deleted checkpoints: {model_checkpoint_dir}")
    except Exception as e:
        print(f"  ✗ Error deleting checkpoints: {str(e)}")


def method_checkpoint_dir(run_folder, model_name):
    """Where per-method weight files go, or None when they are not kept."""
    if not Config.SAVE_METHOD_CHECKPOINTS:
        return None
    return os.path.join(run_folder, model_name, 'checkpoints')


def train_and_evaluate(model_name, loaders, num_classes, class_names, device,
                       run_folder, checkpoints_dir):
    """Train one model, then evaluate Baseline and CWA on the test set."""
    train_loader, val_loader, test_loader, train_labels = loaders

    print(f"\n  [1/3] Training {model_name}...")
    checkpoint_manager, _history = train_model(
        model_name, train_loader, val_loader, num_classes, device,
        class_names=class_names,
        train_labels=train_labels,
        save_dir=run_folder,
        checkpoints_dir=checkpoints_dir,
    )
    print(f"  ✓ Training completed for {model_name}")

    print(f"\n  [2/3] Evaluating {model_name} (Baseline + CWA)...")
    results = evaluate_all_methods(
        model_name,
        checkpoint_manager,
        test_loader,
        train_loader,  # CRITICAL: needed for BatchNorm recalibration
        num_classes,
        device,
        class_names=class_names,
        save_dir=method_checkpoint_dir(run_folder, model_name),
    )
    print(f"  ✓ Evaluation completed for {model_name}")

    return results


def run_cross_validation(splits, num_classes, class_names, device, run_folder):
    """Optional K-Fold mode. The reported tables use multi-seed runs, not CV."""
    train_data, train_labels, val_data, val_labels, test_data, test_labels = splits

    print(f"\n{'='*70}")
    print(f" CROSS-VALIDATION ({Config.CV_N_SPLITS}-Fold Stratified)")
    print(f"{'='*70}")

    # Fold only the official training pool; keep the official test set untouched.
    all_data = concatenate_datasets([train_data, val_data])
    all_labels = train_labels + val_labels

    print(f"  CV pool: {len(all_data)} images")
    print(f"  External test: {len(test_data)} images")

    skf = StratifiedKFold(
        n_splits=Config.CV_N_SPLITS, shuffle=True, random_state=Config.RANDOM_SEED
    )

    all_fold_results = {}

    for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(
        skf.split(np.zeros(len(all_labels)), all_labels), 1
    ):
        print(f"\n{'='*70}")
        print(f" FOLD {fold_idx}/{Config.CV_N_SPLITS}")
        print(f"{'='*70}")

        fold_train_data = all_data.select(fold_train_idx.tolist())
        fold_train_labels = [all_labels[i] for i in fold_train_idx]
        fold_val_data = all_data.select(fold_val_idx.tolist())
        fold_val_labels = [all_labels[i] for i in fold_val_idx]

        print(f"  Train: {len(fold_train_data)} | Val: {len(fold_val_data)} | "
              f"Test (external): {len(test_data)}")

        fold_train_loader, fold_val_loader, fold_test_loader = create_dataloaders(
            fold_train_data, fold_train_labels,
            fold_val_data, test_data,
            Config.BATCH_SIZE, Config.NUM_WORKERS,
        )

        fold_folder = os.path.join(run_folder, f"fold_{fold_idx}")
        os.makedirs(fold_folder, exist_ok=True)

        for model_name in Config.MODELS:
            try:
                fold_ckpt_dir = os.path.join(fold_folder, model_name, 'training_checkpoints')
                results = train_and_evaluate(
                    model_name,
                    (fold_train_loader, fold_val_loader, fold_test_loader, fold_train_labels),
                    num_classes, class_names, device, fold_folder, fold_ckpt_dir,
                )

                all_fold_results.setdefault(model_name, []).append(results)

                print(f"\n  📊 Fold {fold_idx} - {model_name}:")
                for method_name, result in results.items():
                    m = result['metrics']
                    print(f"     {method_name:<30} Acc: {m['Accuracy (%)']:.2f}% | "
                          f"F1: {m['F1-Score (%)']:.2f}% | AUC: {m['AUC (%)']:.2f}%")

                save_model_results(model_name, results, fold_folder)

                if os.path.exists(fold_ckpt_dir):
                    try:
                        shutil.rmtree(fold_ckpt_dir)
                        print(f"    🧹 Deleted training checkpoints: {fold_ckpt_dir}")
                    except Exception as e:
                        print(f"    ⚠ Could not delete training checkpoints: {e}")

                print(f"  ✅ Fold {fold_idx} - {model_name} completed")

            except Exception as e:
                print(f"  ✗ Fold {fold_idx} - {model_name} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

    _export_cv_summary(all_fold_results, run_folder)


def _export_cv_summary(all_fold_results, run_folder):
    """Write mean ± std across folds, per model and method."""
    print(f"\n{'='*70}")
    print(f" CROSS-VALIDATION SUMMARY ({Config.CV_N_SPLITS}-Fold)")
    print(f"{'='*70}")

    summary_rows = []
    for model_name, fold_results_list in all_fold_results.items():
        if not fold_results_list:
            continue
        for method_name in fold_results_list[0]:
            fold_metrics = [fold[method_name]['metrics']
                            for fold in fold_results_list if method_name in fold]
            if not fold_metrics:
                continue

            row = {'Model': model_name, 'Method': method_name,
                   'Folds': len(fold_metrics)}
            for key in METRIC_KEYS:
                values = [m[key] for m in fold_metrics if key in m]
                if values:
                    decimals = 4 if key == 'Test Loss' else 2
                    row[f"{key} (mean)"] = round(float(np.mean(values)), decimals)
                    row[f"{key} (std)"] = round(float(np.std(values)), decimals)
            summary_rows.append(row)

    if not summary_rows:
        print("  No fold results to summarise.")
        return

    df = pd.DataFrame(summary_rows)
    excel_path = os.path.join(run_folder, 'cv_summary.xlsx')
    df.to_excel(excel_path, index=False)

    print(df.to_string(index=False))
    print(f"\n✓ CV summary saved to: {excel_path}")
    print(f"\n{'='*70}")
    print(f" CROSS-VALIDATION COMPLETED!")
    print(f"{'='*70}")


def run_standard(splits, num_classes, class_names, device, run_folder,
                 run_checkpoints_dir):
    """Single train/val/test run — the protocol used for the reported tables."""
    train_data, train_labels, val_data, val_labels, test_data, test_labels = splits

    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, train_labels,
        val_data, test_data,
        Config.BATCH_SIZE, Config.NUM_WORKERS,
    )

    print(f"\n[Step 3/5] Training and evaluating {len(Config.MODELS)} model(s)...")
    print("  Per model: Train → Evaluate (Baseline + CWA) → Save → Clean up")

    # Optional: opens up to 1000 images, so keep disabled for high-compute runs.
    if Config.PRINT_DATASET_STATS:
        print_dataset_statistics(
            concatenate_datasets([train_data, val_data, test_data]),
            train_labels + val_labels + test_labels,
            class_names,
        )

    all_model_results = {}
    successfully_processed = []

    for idx, model_name in enumerate(Config.MODELS, 1):
        print(f"\n{'='*70}")
        print(f"[Model {idx}/{len(Config.MODELS)}] Processing: {model_name}")
        print(f"{'='*70}")

        try:
            results = train_and_evaluate(
                model_name,
                (train_loader, val_loader, test_loader, train_labels),
                num_classes, class_names, device, run_folder, run_checkpoints_dir,
            )
            all_model_results[model_name] = results

            print(f"\n  [3/3] Saving results for {model_name}...")
            save_model_results(model_name, results, run_folder)
            save_confusion_matrices(
                {model_name: results},
                os.path.join(run_folder, model_name),
                class_names=class_names,
            )

            if Config.AUTO_DELETE_CHECKPOINTS:
                print(f"\n  Cleaning up checkpoints for {model_name}...")
                delete_model_checkpoints(model_name, run_checkpoints_dir)
            else:
                print(f"\n  Keeping checkpoints for {model_name} "
                      "(AUTO_DELETE_CHECKPOINTS=False)")

            successfully_processed.append(model_name)
            print(f"\n  ✅ {model_name} completed successfully!")

        except Exception as e:
            print(f"\n  ✗ Error processing {model_name}: {str(e)}")
            print(f"  Skipping {model_name} and continuing with next model...")
            import traceback
            traceback.print_exc()
            continue

    if not all_model_results:
        print("\n✗ No models processed successfully. Exiting...")
        return None

    print(f"\n{'='*70}")
    print(f"✓ Successfully processed {len(successfully_processed)}/{len(Config.MODELS)} models")
    print(f"  Models: {', '.join(successfully_processed)}")
    print(f"{'='*70}")

    print(f"\n[Step 4/5] Combining all results to Excel...")
    excel_path = os.path.join(run_folder, 'all_models_results.xlsx')
    df = export_results_to_excel(all_model_results, excel_path)

    print("\n" + "=" * 70)
    print("COMBINED RESULTS SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))

    print(f"\n[Step 5/5] Generating combined performance chart...")
    create_performance_charts(df, run_folder)

    return df, excel_path


def _print_final_summary(df, excel_path, run_folder, run_number, run_checkpoints_dir):
    print("\n" + "=" * 70)
    print(" RUN COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📊 Run: {run_number}  (dataset: {Config.DATASET}, seed: {Config.RANDOM_SEED})")
    print(f"  - Folder: {run_folder}")
    print(f"  - Combined Excel: {excel_path}")
    print(f"  - Combined Chart: {os.path.join(run_folder, 'performance_comparison.png')}")
    print(f"  - Run Config: {os.path.join(run_folder, 'run_config.xlsx')}")
    print(f"  - Individual Results: {run_folder}/<model_name>/")

    if Config.AUTO_DELETE_CHECKPOINTS:
        print(f"\n💾 Training checkpoints were deleted after evaluation")
    else:
        print(f"\n💾 Training checkpoints kept at: {run_checkpoints_dir}")

    baseline_df = df[df['Method'] == BASELINE_NAME]
    if not baseline_df.empty:
        best_idx = baseline_df['F1-Score (%)'].idxmax()
        print(f"\n🏆 Best Model ({BASELINE_NAME}):")
        print(f"  Model: {baseline_df.loc[best_idx, 'Model']}")
        print(f"  Accuracy: {baseline_df.loc[best_idx, 'Accuracy (%)']:.2f}%")
        print(f"  F1-Score: {baseline_df.loc[best_idx, 'F1-Score (%)']:.2f}%")

    print("\n" + "=" * 70 + "\n")


def main():
    args = parse_args()
    select_dataset(args.dataset)
    apply_cli_overrides(args)

    if args.check_dataset:
        from data import verify_dataset

        print(f"Verifying dataset '{Config.DATASET}'...")
        total, corrupted = verify_dataset()
        print(f"\nChecked {total:,} images, {len(corrupted)} unreadable.")
        for item, error in corrupted[:50]:
            print(f"  {item}: {error}")
        sys.exit(1 if corrupted else 0)

    torch.set_float32_matmul_precision(Config.FLOAT32_MATMUL_PRECISION)

    if run_seed_jobs_if_needed(args):
        return

    print("\n" + "=" * 70)
    print(f" CWA — CLASSIFICATION ({Config.DATASET})")
    print("=" * 70)

    set_seed(Config.RANDOM_SEED, deterministic=args.deterministic)

    print("\n[Step 1/5] Validating configuration...")
    Config.validate_config()

    run_folder, run_number = get_next_run_folder(Config.RESULTS_DIR, args.run_name)
    print(f"\n📁 Run: {run_number}")
    print(f"📁 Results: {run_folder}")

    # Keep training checkpoints isolated per run to allow concurrent terminals.
    if args.checkpoints_dir:
        run_checkpoints_dir = os.path.join(
            Config.CHECKPOINTS_DIR, sanitize_run_name(str(run_number))
        )
    else:
        run_checkpoints_dir = os.path.join(run_folder, "training_checkpoints")
    os.makedirs(run_checkpoints_dir, exist_ok=True)
    print(f"Training checkpoints: {run_checkpoints_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        if Config.USE_AMP and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "USE_AMP=True requires a CUDA GPU with BF16 support for this pipeline"
            )
        print(f"  batch={Config.BATCH_SIZE}, "
              f"precision={Config.AMP_DTYPE if Config.USE_AMP else 'float32'}, "
              f"workers={Config.NUM_WORKERS}")
    else:
        print("⚠ Running in CPU mode. Training will be much slower.")

    print("\n[Step 2/5] Loading and splitting dataset...")
    (train_data, train_labels, val_data, val_labels,
     test_data, test_labels, class_names) = load_dataset(random_seed=Config.RANDOM_SEED)

    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names if num_classes <= 20 else '...'}")

    print("\n[Config] Exporting run configuration...")
    export_run_config(
        run_folder,
        num_classes=num_classes,
        class_names=class_names,
        train_count=len(train_data),
        val_count=len(val_data),
        test_count=len(test_data),
    )

    if Config.USE_WEIGHTED_SAMPLER and Config.LOSS_FUNCTION == 'poly_focal':
        print("\n⚠ WARNING: WeightedRandomSampler and PolyFocalLoss are both ON.")
        print("  Both correct class imbalance (data level vs loss level);")
        print("  consider disabling one if results look off.")

    splits = (train_data, train_labels, val_data, val_labels, test_data, test_labels)

    if Config.USE_CROSS_VALIDATION:
        run_cross_validation(splits, num_classes, class_names, device, run_folder)
        return

    outcome = run_standard(splits, num_classes, class_names, device,
                           run_folder, run_checkpoints_dir)
    if outcome is None:
        return

    df, excel_path = outcome
    _print_final_summary(df, excel_path, run_folder, run_number, run_checkpoints_dir)


if __name__ == "__main__":
    main()
