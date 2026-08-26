"""Run-folder management and seeding.

Every run gets an isolated output folder so concurrent terminals (one per seed
or per model) never overwrite each other.
"""
import json
import os
import random
import re
from datetime import datetime

import numpy as np
import torch

from configs import Config, config_items


def sanitize_run_name(value):
    """Return a filesystem-safe run name."""
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._-") or "run"


def set_seed(seed, deterministic=False):
    """Seed Python, NumPy and torch for reproducibility across all models."""
    print(f"\n🔒 Setting random seeds for reproducibility (seed={seed})...")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # Reproducibility mode is slower and requires CUBLAS_WORKSPACE_CONFIG.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    print("✓ Random seeds set successfully")


def get_next_run_folder(base_results_dir, run_name=None):
    """Create a fresh output folder for this run.

    Folders are claimed with ``os.mkdir`` rather than an exists-check so two
    terminals starting at the same moment cannot land in the same directory.

    With ``run_name``: ``<results>/<name>``, falling back to a
    timestamp+PID suffix if that name is taken. Otherwise auto-increments
    ``<results>/1``, ``<results>/2``, ...
    """
    os.makedirs(base_results_dir, exist_ok=True)

    if run_name:
        safe_name = sanitize_run_name(run_name)
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()

        for candidate in (safe_name, f"{safe_name}_{suffix}_{pid}"):
            run_folder = os.path.join(base_results_dir, candidate)
            try:
                os.mkdir(run_folder)
                return run_folder, candidate
            except FileExistsError:
                continue

        counter = 1
        while True:
            run_id = f"{safe_name}_{suffix}_{pid}_{counter}"
            run_folder = os.path.join(base_results_dir, run_id)
            try:
                os.mkdir(run_folder)
                return run_folder, run_id
            except FileExistsError:
                counter += 1

    existing = [
        int(name) for name in os.listdir(base_results_dir)
        if name.isdigit() and os.path.isdir(os.path.join(base_results_dir, name))
    ]
    next_run = max(existing) + 1 if existing else 1

    while True:
        run_folder = os.path.join(base_results_dir, str(next_run))
        try:
            os.mkdir(run_folder)
            return run_folder, next_run
        except FileExistsError:
            next_run += 1


def export_run_config(run_folder, num_classes=None, class_names=None,
                      train_count=None, val_count=None, test_count=None):
    """Write the full active configuration next to the results.

    Saved as both JSON and Excel so a reviewer can see exactly which settings
    produced a given results folder.
    """
    import pandas as pd

    rows = [
        ("dataset", Config.DATASET),
        ("num_classes", num_classes),
        ("class_names", ", ".join(map(str, class_names)) if class_names else None),
        ("train_images", train_count),
        ("val_images", val_count),
        ("test_images", test_count),
    ]
    rows.extend(Config.describe_source())

    for name, value in config_items():
        if name in ("SUPPORTED_OPTIMIZERS",):
            continue
        rows.append((name.lower(), value))

    # Record what the sampler / mixup flags actually mean for this run.
    rows.append((
        "weighted_sampler",
        "ENABLED (inverse class frequency)" if Config.USE_WEIGHTED_SAMPLER else "DISABLED",
    ))
    rows.append((
        "mixup_cutmix",
        f"mode={Config.MIXUP_MODE}" if Config.USE_MIXUP_CUTMIX else "disabled",
    ))

    serialisable = {key: (value if _is_jsonable(value) else str(value))
                    for key, value in rows}

    json_path = os.path.join(run_folder, "run_config.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, ensure_ascii=False)

    excel_path = os.path.join(run_folder, "run_config.xlsx")
    pd.DataFrame(
        [(key, str(value)) for key, value in rows], columns=["Setting", "Value"]
    ).to_excel(excel_path, index=False)

    print(f"✓ Run config exported to: {json_path}")
    return json_path


def _is_jsonable(value):
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False
