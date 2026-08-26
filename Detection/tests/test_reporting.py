"""Check the detection reporting layer end-to-end with synthetic records.

Verifies method naming, baseline auto-detection, paired deltas, Excel sheets
and charts without needing a trained model or a dataset.

    python tests/test_reporting.py

Exit code 0 means every check passed.
"""
import os
import random
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import reporting as R

LEGACY = re.compile(r"Strategy\s*[123]\b")


class _Box:
    """Stands in for the .box branch of an Ultralytics DetMetrics."""

    def __init__(self, base):
        self.mp, self.mr = base + 0.01, base + 0.02
        self.map50, self.map75, self.map = base + 0.10, base + 0.05, base
        self.ap_class_index = [0, 1]

    def class_result(self, i):
        return (0.6 + i * 0.01, 0.5 + i * 0.01, 0.7 + i * 0.01, 0.55 + i * 0.01)


class _Metrics:
    def __init__(self, base):
        self.box = _Box(base)
        self.fitness = base + 0.01
        self.names = {0: "aeroplane", 1: "bicycle"}
        self.speed = {"inference": 3.1}
        self.save_dir = None


def build_runs(seeds=(1, 10, 42, 100, 500)):
    """One synthetic result set per seed, mirroring run_method_evaluation()."""
    runs = []
    for seed in seeds:
        rng = random.Random(seed)
        records = [
            R.make_strategy_record("Baseline (best ckpt)", "Best", "baseline",
                                   _Metrics(0.500 + rng.random() * 0.01), epochs=[40]),
            R.make_strategy_record("Baseline + BN recal", "Best+BN", "baseline_bn",
                                   _Metrics(0.505 + rng.random() * 0.01),
                                   epochs=[40], bn_update=True),
        ]
        for k in (2, 3, 4, 5):
            records.append(R.make_strategy_record(
                f"CWA (Top-{k})", f"Top-{k}", "topk_avg",
                _Metrics(0.520 + rng.random() * 0.01),
                k=k, epochs=list(range(40, 40 + k)), bn_update=True))
        runs.append({"seed": seed, "strategy_records": records})
    return runs


def main():
    failures = []

    def check(condition, label):
        print(f"  {'PASS' if condition else 'FAIL'}  {label}")
        if not condition:
            failures.append(label)

    tmp = Path(tempfile.mkdtemp(prefix="det_reporting_"))
    summary = R.export_experiment_summary(build_runs(), tmp,
                                          config_snapshot={"model": "yolov8s.pt"})
    check(summary is not None, "SUMMARY.xlsx produced")
    path, mean_std_df, per_seed_df, order, shorts, baseline_name = summary

    check(baseline_name == "Baseline (best ckpt)",
          f"baseline auto-detected as the non-Top-K method ({baseline_name})")
    check(order[0] == "Baseline (best ckpt)", "baseline is first in report order")
    check(len(order) == 6, f"all 6 methods present ({len(order)})")
    check("Method" in per_seed_df.columns and "Strategy" not in per_seed_df.columns,
          "per-seed frame uses the 'Method' column")
    check(per_seed_df["Seed"].nunique() == 5, "all 5 seeds represented")

    R.export_experiment_charts(mean_std_df, per_seed_df, order, shorts,
                               baseline_name, tmp / "charts")
    charts = sorted(p.name for p in (tmp / "charts").glob("*.png"))
    check(len(charts) == 5, f"5 summary charts rendered: {charts}")

    sheets = pd.read_excel(path, sheet_name=None)
    check(set(sheets) >= {"Mean_Std", "PerSeed", "Delta_vs_Baseline"},
          f"expected sheets present: {list(sheets)}")

    leaks = []
    for name, df in sheets.items():
        leaks += [(name, c) for c in df.columns if LEGACY.search(str(c))]
        for column in df.columns:
            for value in df[column].tolist():
                # Absolute paths may legitimately contain unrelated words.
                if isinstance(value, str) and LEGACY.search(value) and not os.path.isabs(value):
                    leaks.append((name, column, value))
    check(not leaks, f"no legacy 'Strategy N' labels in any sheet ({leaks})")

    delta = sheets["Delta_vs_Baseline"]
    check(len(delta) == 5 and set(delta["Baseline"]) == {"Baseline (best ckpt)"},
          "paired deltas computed against the baseline for every other method")

    print()
    if failures:
        print(f" {len(failures)} CHECK(S) FAILED")
        return 1
    print(" DETECTION REPORTING OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
