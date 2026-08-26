"""Check the segmentation reporting layer end-to-end with synthetic results.

Verifies method naming, baseline resolution, paired deltas, Excel sheets and
charts without needing a trained model or a dataset.

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

import evaluate as E

LEGACY = re.compile(r"Strategy\s*[123]\b")


class _Branch:
    """Stands in for the .seg / .box branch of an Ultralytics SegmentMetrics."""

    def __init__(self, base):
        self.mp, self.mr = base + 0.01, base + 0.02
        self.map50, self.map75, self.map = base + 0.10, base + 0.05, base
        self.ap_class_index = [0, 1]

    def class_result(self, i):
        return (0.6 + i * 0.01, 0.5 + i * 0.01, 0.7 + i * 0.01, 0.55 + i * 0.01)


class _Metrics:
    def __init__(self, base):
        self.seg = _Branch(base)
        self.box = _Branch(base - 0.01)
        self.fitness = base + 0.01
        self.names = {0: "wheel", 1: "door"}
        self.speed = {"inference": 4.2}


def build_runs(seeds=(1, 10, 42, 100, 500)):
    """One synthetic result set per seed, mirroring run_method_evaluation()."""
    runs = []
    for seed in seeds:
        rng = random.Random(seed)
        results = {
            E.BASELINE_NAME: _Metrics(0.500 + rng.random() * 0.01),
            E.TOP1_NAME: _Metrics(0.505 + rng.random() * 0.01),
        }
        for k in (2, 3, 4, 5):
            results[E.cwa_name(k)] = _Metrics(0.520 + rng.random() * 0.01)
        runs.append({"seed": seed, "run_dir": f"/runs/seed{seed}",
                     "strategy_results": results})
    return runs


def main():
    failures = []

    def check(condition, label):
        print(f"  {'PASS' if condition else 'FAIL'}  {label}")
        if not condition:
            failures.append(label)

    tmp = Path(tempfile.mkdtemp(prefix="seg_reporting_"))
    out = E.export_multi_seed_summary(build_runs(), tmp, exp_name="carparts_test")

    check(out["excel"] is not None, "summary workbook produced")
    check(len(out["charts"]) == 6, f"6 summary charts rendered ({len(out['charts'])})")

    methods = list(out["mean_std"]["Method"])
    check(methods[0] == E.BASELINE_NAME, f"baseline reported first ({methods[0]})")
    check(len(methods) == 6, f"all 6 methods present: {methods}")
    check(E.cwa_name(3) == "CWA (Top-3 avg)", f"CWA label format: {E.cwa_name(3)}")

    sheets = pd.read_excel(out["excel"], sheet_name=None)
    check(set(sheets) >= {"MeanStd", "DeltaVsBaseline", "PerSeed"},
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

    check(len(sheets["PerSeed"]) == 30, "PerSeed has 5 seeds x 6 methods")
    check(len(sheets["DeltaVsBaseline"]) == 5,
          "paired deltas computed for every non-baseline method")

    # Chart labelling must still parse K out of the renamed methods.
    import charts as C
    check(C.strategy_k(E.BASELINE_NAME) is None, "baseline carries no K")
    check(C.strategy_k(E.TOP1_NAME) == 1, "Top-1 baseline parses as K=1")
    check([C.strategy_k(E.cwa_name(k)) for k in (2, 3, 4, 5)] == [2, 3, 4, 5],
          "CWA labels parse back to their K")

    print()
    if failures:
        print(f" {len(failures)} CHECK(S) FAILED")
        return 1
    print(" SEGMENTATION REPORTING OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
