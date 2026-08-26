"""
Reporting layer cho CWA - Object Detection (Ultralytics YOLO + Pascal VOC).

Tách khỏi evaluate.py để mỗi file một việc:
  - evaluate.py : ĐO (model.val, rank checkpoint, average, BN recalibration)
  - reporting.py: TRÌNH BÀY (trích metrics, Excel, chart, dọn cây thư mục)

Cây thư mục kết quả một experiment (`python main.py train --exp-name <NAME>`):

    results/detection/<NAME>/
    ├── SUMMARY.xlsx              # ★ mean ± std của TẤT CẢ seed × strategy
    ├── experiment_config.json    # snapshot config lúc chạy (provenance)
    ├── charts/                   # chart tổng hợp cả experiment
    │   ├── 01_topk_curve_mAP50-95.png
    │   ├── 02_method_mAP50-95.png
    │   ├── 03_method_mAP50.png
    │   ├── 04_delta_vs_baseline.png
    │   └── 05_per_seed_paired.png
    └── seeds/
        ├── seed_1/
        │   ├── results_seed_1.xlsx
        │   └── charts/
        │       ├── training/          # curve + confusion matrix trên val'
        │       └── test_<method>/   # PR/F1 curve + confusion matrix trên test
        └── seed_10/ ...

KHÔNG có checkpoint (.pt) nào được giữ lại: chúng chỉ là file tạm phục vụ
ranking/averaging và bị xóa ngay sau khi evaluate xong.
"""
import json
import math
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import Config

# ==================== Palette (validated, light surface #ffffff) ====================
# Nguồn: bảng màu tham chiếu đã qua validator (lightness band, chroma floor,
# CVD separation, contrast). Chart cho paper in trên nền trắng nên dùng light mode.
INK = "#0b0b0b"          # text chính
INK_SOFT = "#52514e"     # text phụ / đường baseline tham chiếu
MUTED = "#898781"        # nhãn trục
GRID = "#e1e0d9"         # gridline mảnh
AXIS = "#c3c2b7"         # trục
POS = "#2a78d6"          # cực dương (cải thiện) của thang diverging
NEG = "#e34948"          # cực âm (tệ đi)
# Ordinal ramp 1 hue (blue), light→dark — dùng cho K vì K là đại lượng có thứ tự.
ORDINAL_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

# Thứ tự cột metric dùng xuyên suốt Excel/chart
METRIC_COLUMNS = ["mAP@0.5:0.95", "mAP@0.5", "mAP@0.75", "Precision", "Recall", "Fitness"]
PRIMARY_METRIC = "mAP@0.5:0.95"

INDEPENDENT_VAL_NOTE = (
    "val' ĐỘC LẬP tách từ train theo VAL_RATIO — chỉ dùng để chọn best checkpoint / "
    "rank Top-K. Test = VOC2007 test (4952 ảnh) giữ nguyên làm hold-out, chỉ dùng "
    "cho báo cáo cuối."
)
VAL_TEST_WARNING = (
    "VAL_RATIO=0: dùng nguyên data.yaml gốc. Với VOC.yaml của Ultralytics, split 'val' "
    "và 'test' TRÙNG NHAU (đều là VOC2007 test) — không có val độc lập, checkpoint được "
    "chọn trên chính tập test (leakage)."
)


# ==================== Metrics extraction (từ DetMetrics) ====================

class MetricsSnapshot:
    """
    Bản chụp CHỈ-SỐ-LIỆU của một ``DetMetrics``.

    Vì sao không giữ thẳng DetMetrics: object đó giữ ``on_plot`` = bound method
    của validator, kéo theo ``validator.dataloader`` và toàn bộ worker process
    của dataloader (xem memory.py). Kết quả từng seed được giữ tới cuối
    experiment, nên giữ DetMetrics = giữ luôn mọi dataloader của mọi lượt eval
    → RAM cộng dồn qua các seed và job bị OOM killer giết.

    Snapshot chỉ gồm float/str/dict thuần nên an toàn để giữ bao lâu tùy ý.
    """

    __slots__ = ("overall", "per_class", "speed", "save_dir")

    def __init__(self, overall, per_class, speed=None, save_dir=None):
        self.overall = dict(overall or {})
        self.per_class = [dict(row) for row in (per_class or [])]
        self.speed = dict(speed or {})
        self.save_dir = save_dir

    def __repr__(self):
        primary = self.overall.get("mAP@0.5:0.95")
        return f"MetricsSnapshot(mAP@0.5:0.95={primary}, per_class={len(self.per_class)})"


def snapshot_metrics(metrics):
    """
    DetMetrics → MetricsSnapshot (idempotent: snapshot vào thì snapshot ra).

    Gọi NGAY sau ``model.val()`` rồi vứt DetMetrics đi.
    """
    if metrics is None or isinstance(metrics, MetricsSnapshot):
        return metrics
    return MetricsSnapshot(
        overall=extract_overall_metrics(metrics),
        per_class=extract_per_class_metrics(metrics),
        speed=getattr(metrics, "speed", None),
        save_dir=getattr(metrics, "save_dir", None),
    )


def extract_overall_metrics(metrics):
    """Dict metrics overall từ DetMetrics của Ultralytics (metrics.box.*)."""
    if isinstance(metrics, MetricsSnapshot):
        return dict(metrics.overall)
    box = metrics.box
    overall = {
        "Precision": float(box.mp),
        "Recall": float(box.mr),
        "mAP@0.5": float(box.map50),
        "mAP@0.75": float(box.map75),
        "mAP@0.5:0.95": float(box.map),
    }
    # fitness = 0.1*mAP50 + 0.9*mAP50-95 (định nghĩa của Ultralytics)
    fitness = getattr(metrics, "fitness", None)
    if fitness is not None:
        overall["Fitness"] = float(fitness)
    return overall


def extract_per_class_metrics(metrics):
    """
    List dict per-class (P, R, AP@0.5, AP@0.5:0.95) từ DetMetrics.

    box.ap_class_index = các class-id thực sự xuất hiện trong tập eval;
    box.class_result(i) trả (p, r, ap50, ap) cho phần tử thứ i.
    """
    if isinstance(metrics, MetricsSnapshot):
        return [dict(row) for row in metrics.per_class]
    box = metrics.box
    names = getattr(metrics, "names", {}) or {}
    rows = []
    for i, class_idx in enumerate(getattr(box, "ap_class_index", [])):
        class_idx = int(class_idx)
        p, r, ap50, ap = box.class_result(i)
        rows.append({
            "Class ID": class_idx,
            "Class": str(names.get(class_idx, class_idx)),
            "Precision": float(p),
            "Recall": float(r),
            "AP@0.5": float(ap50),
            "AP@0.5:0.95": float(ap),
        })
    return rows


def print_detection_metrics(metrics, header="DETECTION EVALUATION RESULTS"):
    """In metrics detection ra console."""
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


# ==================== Strategy records ====================
# Một "record" mô tả đầy đủ 1 dòng kết quả: tên, nhãn ngắn cho chart, K, các
# epoch nguồn, có BN recalibration hay không, và DetMetrics tương ứng.

def make_strategy_record(name, short, kind, metrics, k=None, epochs=None, bn_update=False):
    # metrics luôn được chụp thành MetricsSnapshot: record này sống tới cuối
    # experiment, giữ DetMetrics gốc là giữ luôn validator + dataloader của nó.
    return {
        "name": name,
        "short": short,
        "kind": kind,          # "baseline" | "baseline_bn" | "topk_avg"
        "k": k,
        "epochs": list(epochs or []),
        "bn_update": bool(bn_update),
        "metrics": snapshot_metrics(metrics),
    }


def normalize_strategy_records(strategy_results):
    """
    Chấp nhận cả list[record] lẫn dict {name: DetMetrics} (đường `main.py eval`)
    và luôn trả về list[record].
    """
    if strategy_results is None:
        return []
    if isinstance(strategy_results, dict):
        return [
            make_strategy_record(name, name, "unknown", metrics)
            for name, metrics in strategy_results.items()
            if metrics is not None
        ]
    return [r for r in strategy_results if r.get("metrics") is not None]


def slugify(text):
    """Tên file/thư mục an toàn từ tên strategy ('CWA (Top-3)' → 'cwa_top_3')."""
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_").lower()
    return slug or "item"


# ==================== Thống kê mean/std + paired t-test ====================

def _mean(values):
    return sum(values) / len(values) if values else float("nan")


def _std(values):
    """Sample standard deviation (ddof=1) — quy ước báo cáo mean ± std trên N seed."""
    n = len(values)
    if n < 2:
        return float("nan")
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (n - 1))


def _betacf(a, b, x, max_iter=200, eps=3e-16):
    """Continued fraction cho incomplete beta (thuật toán Lentz, Numerical Recipes)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b) — chỉ cần math, không cần scipy."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(log_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + b * math.log(1.0 - x) + a * math.log(x)
    ) * _betacf(b, a, 1.0 - x) / b


def paired_t_test(deltas):
    """
    Paired two-sided t-test trên list chênh lệch per-seed (H0: mean = 0).

    Dùng cho so sánh CWA vs baseline: cùng một seed → cùng data split và
    cùng quá trình train, nên đây là thiết kế paired (đúng hơn unpaired t-test).

    Returns:
        (t_stat, p_value, df) — (nan, nan, n-1) khi không đủ mẫu hoặc std = 0.
    """
    n = len(deltas)
    if n < 2:
        return float("nan"), float("nan"), max(n - 1, 0)
    mean_d = _mean(deltas)
    std_d = _std(deltas)
    if std_d == 0 or not math.isfinite(std_d):
        return float("nan"), float("nan"), n - 1
    t_stat = mean_d / (std_d / math.sqrt(n))
    df = n - 1
    p_value = _betainc(df / 2.0, 0.5, df / (df + t_stat * t_stat))
    return t_stat, p_value, df


def format_mean_std(mean, std, decimals=4):
    """'0.7712 ± 0.0031' — chuỗi copy thẳng vào bảng của paper."""
    if mean is None or not math.isfinite(mean):
        return "n/a"
    if std is None or not math.isfinite(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


# ==================== Excel: per-seed ====================

def read_results_csv(run_dir):
    """Đọc results.csv Ultralytics sinh trong run_dir → DataFrame per-epoch."""
    csv_path = os.path.join(str(run_dir), "results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"results.csv not found in run dir: {run_dir}")
    df = pd.read_csv(csv_path)
    # Bản Ultralytics cũ pad khoảng trắng trong header → chuẩn hóa tên cột
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _data_note(data):
    """Chọn note đúng theo cách chuẩn bị data (holdout hay data.yaml gốc)."""
    if "holdout" in os.path.basename(str(data)):
        return INDEPENDENT_VAL_NOTE
    return VAL_TEST_WARNING


def _run_info_rows(run_dir, data, split, seed, summary_source):
    try:
        import ultralytics
        ultralytics_version = ultralytics.__version__
    except ImportError:
        ultralytics_version = "N/A"

    return [
        ("model", Config.MODEL or "N/A"),
        ("data", str(data)),
        ("val_ratio (tách val' từ train)", Config.VAL_RATIO),
        ("epochs", Config.EPOCHS),
        ("imgsz", Config.IMGSZ),
        ("batch", Config.BATCH),
        ("optimizer", Config.OPTIMIZER),
        ("lr0 / lrf", f"{Config.LR0} / {Config.LRF}"),
        ("warmup_epochs / cos_lr", f"{Config.WARMUP_EPOCHS} / {Config.COS_LR}"),
        ("mixup / copy_paste", f"{Config.MIXUP} / {Config.COPY_PASTE}"),
        ("patience (early stopping)", Config.PATIENCE),
        ("device", Config.DEVICE if Config.DEVICE is not None else "auto"),
        ("seed", seed if seed is not None else Config.RANDOM_SEED),
        ("cls_loss", str(Config.LOSS_FUNCTION) + (
            f" (gamma={Config.FOCAL_GAMMA}, alpha={Config.FOCAL_ALPHA})"
            if str(Config.LOSS_FUNCTION).lower() == "focal" else ""
        )),
        ("eval_split (báo cáo cuối)", split or Config.EVAL_SPLIT or "mặc định theo data.yaml"),
        ("cwa top_k_values", str(Config.TOP_K_VALUES) if Config.USE_CWA else "OFF"),
        ("averaging", "uniform element-wise mean của learnable parameters (KHÔNG EMA)"),
        ("bn_recalibration", (
            f"ON — {Config.BN_UPDATE_BATCHES} batch trên split train, "
            f"momentum=None (cumulative), no_grad"
            if Config.USE_BN_UPDATE else "OFF"
        )),
        ("checkpoint_weight_source", "raw (EMA disabled)" if Config.USE_CWA else "Ultralytics default"),
        ("checkpoint_retention", "temporary; deleted after evaluation"
            if Config.DELETE_CHECKPOINTS_AFTER_RUN else "kept"),
        ("run_dir", str(run_dir)),
        ("summary_source", summary_source),
        ("export_date", datetime.now().isoformat(timespec="seconds")),
        ("ultralytics_version", ultralytics_version),
        ("NOTE data split", _data_note(data)),
    ]


def _overall_dataframe(records):
    rows = []
    for record in records:
        row = {
            "Method": record["name"],
            "K": record.get("k"),
            "BN recal": "yes" if record.get("bn_update") else "no",
            "Source epochs": ", ".join(str(e) for e in record.get("epochs") or []) or "-",
        }
        row.update(extract_overall_metrics(record["metrics"]))
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    lead = ["Method", "K", "BN recal", "Source epochs"]
    ordered = lead + [c for c in METRIC_COLUMNS if c in df.columns]
    return df[ordered + [c for c in df.columns if c not in ordered]]


def _per_class_dataframe(records):
    rows = []
    for record in records:
        for row in extract_per_class_metrics(record["metrics"]):
            rows.append({"Method": record["name"], **row})
    return pd.DataFrame(rows)


def _ranking_dataframe(ranking, top_k_values):
    """Bảng Top-K checkpoint: epoch nào, fitness val' bao nhiêu, dùng cho K nào."""
    rows = []
    k_values = sorted(int(k) for k in (top_k_values or []))
    for rank, item in enumerate(ranking or [], start=1):
        used_in = [str(k) for k in k_values if rank <= k]
        rows.append({
            "Rank": rank,
            "Epoch": item.get("epoch"),
            "Val fitness": item.get("fitness"),
            "Checkpoint file": item.get("file"),
            "Dùng trong Top-K": ", ".join(used_in) or "-",
        })
    return pd.DataFrame(rows)


def _autosize(writer, sheet_name, df, offset=0):
    """Giãn cột theo nội dung để Excel đọc được ngay, không phải kéo tay."""
    try:
        worksheet = writer.sheets[sheet_name]
    except (AttributeError, KeyError):
        return
    for idx, column in enumerate(df.columns):
        values = [str(column)] + [str(v) for v in df[column].head(200).tolist()]
        width = min(max(len(v) for v in values) + 2, 60)
        worksheet.column_dimensions[
            worksheet.cell(row=1, column=idx + 1 + offset).column_letter
        ].width = width


def export_seed_excel(
    run_dir,
    strategy_results=None,
    data=None,
    split=None,
    output_path=None,
    seed=None,
    ranking=None,
):
    """
    Xuất Excel kết quả của MỘT seed.

    Sheets:
      - RunInfo          : toàn bộ hyperparameter + note về data split
      - Overall          : 1 dòng / strategy (P, R, mAP50, mAP75, mAP50-95, fitness)
      - PerClass         : AP từng class × strategy
      - TopK_Checkpoints : epoch nào lọt Top-K, fitness val' bao nhiêu
      - PerEpoch         : nguyên results.csv của Ultralytics (mỗi epoch 1 dòng)
    """
    run_dir = str(run_dir)
    data = data or Config.DATA
    records = normalize_strategy_records(strategy_results)

    per_epoch_df = None
    try:
        per_epoch_df = read_results_csv(run_dir)
    except FileNotFoundError:
        print(f"  ⚠ Không tìm thấy results.csv trong {run_dir} — bỏ qua sheet PerEpoch")

    if records:
        overall_df = _overall_dataframe(records)
        per_class_df = _per_class_dataframe(records)
        summary_source = "model.val() — Ultralytics DetMetrics"
    elif per_epoch_df is not None:
        last = per_epoch_df.iloc[-1]
        column_map = {
            "Precision": "metrics/precision(B)",
            "Recall": "metrics/recall(B)",
            "mAP@0.5": "metrics/mAP50(B)",
            "mAP@0.5:0.95": "metrics/mAP50-95(B)",
        }
        row = {"Method": "Last epoch (results.csv)"}
        for metric_name, column in column_map.items():
            if column in per_epoch_df.columns:
                row[metric_name] = float(last[column])
        overall_df = pd.DataFrame([row])
        per_class_df = pd.DataFrame()
        summary_source = "results.csv (epoch cuối) — chạy `strategies`/`eval` để có per-class AP"
    else:
        raise ValueError("Không có strategy results lẫn results.csv — không thể export Excel")

    if output_path is None:
        default_name = f"results_seed_{seed}.xlsx" if seed is not None else "results.xlsx"
        output_path = os.path.join(run_dir, default_name)
    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    info_df = pd.DataFrame(
        _run_info_rows(run_dir, data, split, seed, summary_source),
        columns=["Parameter", "Value"],
    )
    ranking_df = _ranking_dataframe(ranking, Config.TOP_K_VALUES)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        info_df.to_excel(writer, sheet_name="RunInfo", index=False)
        _autosize(writer, "RunInfo", info_df)

        overall_df.to_excel(writer, sheet_name="Overall", index=False)
        _autosize(writer, "Overall", overall_df)

        if not per_class_df.empty:
            per_class_df.to_excel(writer, sheet_name="PerClass", index=False)
            _autosize(writer, "PerClass", per_class_df)

        if not ranking_df.empty:
            ranking_df.to_excel(writer, sheet_name="TopK_Checkpoints", index=False)
            _autosize(writer, "TopK_Checkpoints", ranking_df)

        if per_epoch_df is not None:
            per_epoch_df.to_excel(writer, sheet_name="PerEpoch", index=False)

    print(f"\n  ✓ Excel{f' (seed {seed})' if seed is not None else ''}: {output_path}")
    print(f"    Overall {len(overall_df)} strategy · PerClass {len(per_class_df)} dòng · "
          f"PerEpoch {0 if per_epoch_df is None else len(per_epoch_df)} epoch")
    return output_path


# ==================== Excel: tổng hợp toàn experiment ====================

def _build_summary_frames(all_runs_results):
    """
    Từ list kết quả các seed → (per_seed_df, per_class_df, ranking_df, order, shorts).

    `order` giữ đúng thứ tự strategy xuất hiện lần đầu để bảng và chart nhất
    quán; `shorts` map tên đầy đủ → nhãn ngắn dùng trên trục chart.
    """
    per_seed_rows, per_class_rows, ranking_rows = [], [], []
    order, shorts = [], {}

    for run in all_runs_results:
        seed = run.get("seed")
        records = normalize_strategy_records(run.get("strategy_records") or run.get("strategy_results"))
        for record in records:
            name = record["name"]
            if name not in order:
                order.append(name)
                shorts[name] = record.get("short") or name
            per_seed_rows.append({
                "Seed": seed,
                "Method": name,
                "K": record.get("k"),
                "BN recal": "yes" if record.get("bn_update") else "no",
                "Source epochs": ", ".join(str(e) for e in record.get("epochs") or []) or "-",
                **extract_overall_metrics(record["metrics"]),
            })
            for row in extract_per_class_metrics(record["metrics"]):
                per_class_rows.append({"Seed": seed, "Method": name, **row})

        for rank, item in enumerate(run.get("ranking") or [], start=1):
            ranking_rows.append({
                "Seed": seed,
                "Rank": rank,
                "Epoch": item.get("epoch"),
                "Val fitness": item.get("fitness"),
            })

    return (
        pd.DataFrame(per_seed_rows),
        pd.DataFrame(per_class_rows),
        pd.DataFrame(ranking_rows),
        order,
        shorts,
    )


def _mean_std_frame(per_seed_df, order, shorts):
    """Bảng mean ± std của từng strategy trên tất cả seed."""
    metric_cols = [c for c in METRIC_COLUMNS if c in per_seed_df.columns]
    rows = []
    for name in order:
        group = per_seed_df[per_seed_df["Method"] == name]
        if group.empty:
            continue
        row = {
            "Method": name,
            "Label": shorts.get(name, name),
            "N seeds": int(len(group)),
            "K": group["K"].iloc[0] if "K" in group.columns else None,
            "BN recal": group["BN recal"].iloc[0] if "BN recal" in group.columns else None,
        }
        for col in metric_cols:
            values = [float(v) for v in group[col].tolist() if pd.notna(v)]
            mean, std = _mean(values), _std(values)
            row[f"{col} mean"] = mean
            row[f"{col} std"] = std
            row[f"{col} (mean ± std)"] = format_mean_std(mean, std)
        rows.append(row)
    return pd.DataFrame(rows)


def _delta_frame(per_seed_df, order, baseline_name):
    """
    So sánh paired từng strategy với baseline trên CÙNG seed.

    Mỗi seed cho một cặp (baseline, strategy) → delta. Báo cáo mean/std của
    delta, số seed thắng, t-statistic và p-value (paired two-sided t-test).
    """
    if baseline_name is None or "Seed" not in per_seed_df.columns:
        return pd.DataFrame()

    metric_cols = [c for c in (PRIMARY_METRIC, "mAP@0.5") if c in per_seed_df.columns]
    baseline = per_seed_df[per_seed_df["Method"] == baseline_name].set_index("Seed")
    rows = []
    for name in order:
        if name == baseline_name:
            continue
        group = per_seed_df[per_seed_df["Method"] == name].set_index("Seed")
        shared = [s for s in group.index if s in baseline.index]
        if not shared:
            continue
        row = {"Method": name, "Baseline": baseline_name, "N paired seeds": len(shared)}
        for col in metric_cols:
            deltas = [float(group.loc[s, col]) - float(baseline.loc[s, col]) for s in shared]
            t_stat, p_value, df_dof = paired_t_test(deltas)
            row[f"Δ {col} mean"] = _mean(deltas)
            row[f"Δ {col} std"] = _std(deltas)
            row[f"Δ {col} (mean ± std)"] = format_mean_std(_mean(deltas), _std(deltas))
            row[f"Δ {col} wins"] = f"{sum(1 for d in deltas if d > 0)}/{len(deltas)}"
            row[f"Δ {col} t"] = t_stat
            row[f"Δ {col} p (paired, 2-sided)"] = p_value
            row[f"Δ {col} df"] = df_dof
        rows.append(row)
    return pd.DataFrame(rows)


def _per_class_mean_std_frame(per_class_df, order):
    """AP trung bình ± std của từng class × strategy trên tất cả seed."""
    if per_class_df.empty:
        return pd.DataFrame()
    rows = []
    for name in order:
        group = per_class_df[per_class_df["Method"] == name]
        if group.empty:
            continue
        for class_name, class_group in group.groupby("Class", sort=False):
            row = {
                "Method": name,
                "Class ID": int(class_group["Class ID"].iloc[0]),
                "Class": class_name,
                "N seeds": int(len(class_group)),
            }
            for col in ("AP@0.5", "AP@0.5:0.95"):
                values = [float(v) for v in class_group[col].tolist() if pd.notna(v)]
                row[f"{col} mean"] = _mean(values)
                row[f"{col} std"] = _std(values)
                row[f"{col} (mean ± std)"] = format_mean_std(_mean(values), _std(values))
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(["Class ID", "Method"]) if not df.empty else df


def _config_frame(config_snapshot):
    rows = [(k, json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
            for k, v in (config_snapshot or {}).items()]
    return pd.DataFrame(rows, columns=["Parameter", "Value"])


def export_experiment_summary(all_runs_results, exp_dir, config_snapshot=None, baseline_name=None):
    """
    Xuất SUMMARY.xlsx — file tổng hợp mean ± std của TẤT CẢ seed × strategy.

    Sheets:
      - Mean_Std           : ★ 1 dòng / strategy, có sẵn cột "mean ± std" định dạng
      - PerSeed            : 1 dòng / (seed × strategy) — số liệu thô
      - Delta_vs_Baseline  : paired delta vs baseline + t-test
      - PerClass_Mean_Std  : AP từng class, mean ± std trên các seed
      - PerClass_PerSeed   : AP từng class, số liệu thô
      - TopK_Checkpoints   : epoch nào lọt Top-K ở mỗi seed
      - Config             : snapshot config lúc chạy

    Returns:
        (output_path, mean_std_df, per_seed_df, order, shorts) hoặc None nếu rỗng.
    """
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    output_path = exp_dir / "SUMMARY.xlsx"

    per_seed_df, per_class_df, ranking_df, order, shorts = _build_summary_frames(all_runs_results)
    if per_seed_df.empty:
        print("  ⚠ Không có kết quả nào để export SUMMARY.xlsx")
        return None

    if baseline_name is None:
        baseline_name = next(
            (n for n in order if n in per_seed_df["Method"].values and "Top-" not in n),
            order[0] if order else None,
        )

    mean_std_df = _mean_std_frame(per_seed_df, order, shorts)
    delta_df = _delta_frame(per_seed_df, order, baseline_name)
    per_class_ms_df = _per_class_mean_std_frame(per_class_df, order)
    config_df = _config_frame(config_snapshot)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        mean_std_df.to_excel(writer, sheet_name="Mean_Std", index=False)
        _autosize(writer, "Mean_Std", mean_std_df)

        per_seed_df.to_excel(writer, sheet_name="PerSeed", index=False)
        _autosize(writer, "PerSeed", per_seed_df)

        if not delta_df.empty:
            delta_df.to_excel(writer, sheet_name="Delta_vs_Baseline", index=False)
            _autosize(writer, "Delta_vs_Baseline", delta_df)

        if not per_class_ms_df.empty:
            per_class_ms_df.to_excel(writer, sheet_name="PerClass_Mean_Std", index=False)
            _autosize(writer, "PerClass_Mean_Std", per_class_ms_df)
        if not per_class_df.empty:
            per_class_df.to_excel(writer, sheet_name="PerClass_PerSeed", index=False)

        if not ranking_df.empty:
            ranking_df.to_excel(writer, sheet_name="TopK_Checkpoints", index=False)
            _autosize(writer, "TopK_Checkpoints", ranking_df)

        if not config_df.empty:
            config_df.to_excel(writer, sheet_name="Config", index=False)
            _autosize(writer, "Config", config_df)

    print(f"\n  ✓ SUMMARY.xlsx: {output_path}")
    print(f"    Mean_Std {len(mean_std_df)} strategy × {per_seed_df['Seed'].nunique()} seed")
    return output_path, mean_std_df, per_seed_df, order, shorts, baseline_name


# ==================== Charts ====================

def _init_matplotlib():
    """Import matplotlib ở backend không cần màn hình; None nếu chưa cài."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠ matplotlib chưa được cài — bỏ qua phần vẽ chart")
        return None
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.titleweight": "bold",
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SOFT,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
    })
    return plt


def _style_axes(ax, ylabel=None, title=None, xlabel=None):
    ax.grid(axis="y", linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    if title:
        ax.set_title(title, color=INK, pad=12, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SOFT)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SOFT)


def _ordinal_colors(n):
    """n màu từ ramp 1-hue light→dark (K là đại lượng có thứ tự, không phải nhãn)."""
    if n <= 0:
        return []
    if n == 1:
        return [ORDINAL_BLUE[2]]
    step = (len(ORDINAL_BLUE) - 1) / (n - 1)
    return [ORDINAL_BLUE[min(int(round(i * step)), len(ORDINAL_BLUE) - 1)] for i in range(n)]


def _chart_topk_curve(plt, mean_std_df, baseline_name, metric, out_path):
    """
    Hình chính của paper: mAP theo K, dải ±std, kèm đường tham chiếu baseline.

    Form: K là biến có thứ tự → line + band (không phải bar). Một series duy
    nhất nên không cần legend box; baseline được direct-label ngay trên đường.
    """
    topk = mean_std_df[mean_std_df["K"].notna() & (mean_std_df["Method"] != baseline_name)]
    topk = topk[topk["K"].astype(float) > 1].sort_values("K")
    if topk.empty:
        return None

    ks = [int(k) for k in topk["K"].tolist()]
    means = [float(v) for v in topk[f"{metric} mean"].tolist()]
    stds = [0.0 if not math.isfinite(float(v)) else float(v) for v in topk[f"{metric} std"].tolist()]

    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    base_row = mean_std_df[mean_std_df["Method"] == baseline_name]
    if not base_row.empty:
        base_mean = float(base_row[f"{metric} mean"].iloc[0])
        base_std = float(base_row[f"{metric} std"].iloc[0])
        if math.isfinite(base_std) and base_std > 0:
            ax.axhspan(base_mean - base_std, base_mean + base_std, color=INK_SOFT, alpha=0.08, lw=0)
        ax.axhline(base_mean, color=INK_SOFT, ls="--", lw=1.4)
        ax.annotate(
            f"baseline {base_mean:.4f}",
            xy=(ks[0], base_mean), xytext=(0, 5), textcoords="offset points",
            color=INK_SOFT, fontsize=9, va="bottom",
        )

    lower = [m - s for m, s in zip(means, stds)]
    upper = [m + s for m, s in zip(means, stds)]
    ax.fill_between(ks, lower, upper, color=POS, alpha=0.14, lw=0)
    ax.plot(ks, means, color=POS, lw=2.0, marker="o", markersize=7,
            markerfacecolor=POS, markeredgecolor="#ffffff", markeredgewidth=1.6)

    best_idx = max(range(len(means)), key=lambda i: means[i])
    for i, (k, m) in enumerate(zip(ks, means)):
        weight = "bold" if i == best_idx else "normal"
        ax.annotate(f"{m:.4f}", xy=(k, m), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=9, color=INK, fontweight=weight)

    ax.set_xticks(ks)
    ax.set_xticklabels([f"K={k}" for k in ks])
    # Chừa lề ngang để nhãn giá trị ở K đầu/cuối không bị cắt mất
    ax.set_xlim(ks[0] - 0.4, ks[-1] + 0.4)
    _style_axes(ax, ylabel=metric, xlabel="Số checkpoint được average (K)",
                title=f"Top-K checkpoint averaging — {metric} (mean ± std trên các seed)")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _chart_strategy_dots(plt, mean_std_df, baseline_name, metric, out_path):
    """
    Dot plot ngang: điểm = mean, thanh ngang = ±std, mỗi strategy một dòng.

    Cố ý KHÔNG dùng bar: chênh lệch giữa các strategy chỉ cỡ 1e-3 trên thang
    0–1, nên bar buộc phải cắt trục y khỏi 0 — mà bar cắt gốc thì chiều dài cột
    không còn tỉ lệ với giá trị và thổi phồng khác biệt. Dot plot không mã hóa
    diện-tích-từ-0 nên đọc đúng ở mọi khoảng trục.
    """
    if mean_std_df.empty or f"{metric} mean" not in mean_std_df.columns:
        return None

    labels = mean_std_df["Label"].tolist()
    means = [float(v) for v in mean_std_df[f"{metric} mean"].tolist()]
    stds = [0.0 if not math.isfinite(float(v)) else float(v)
            for v in mean_std_df[f"{metric} std"].tolist()]

    topk_mask = [
        bool(pd.notna(k)) and float(k) > 1 and name != baseline_name
        for k, name in zip(mean_std_df["K"], mean_std_df["Method"])
    ]
    ramp = _ordinal_colors(sum(topk_mask))
    colors, ramp_i = [], 0
    for is_topk in topk_mask:
        colors.append(ramp[ramp_i] if is_topk else INK_SOFT)
        ramp_i += 1 if is_topk else 0

    fig, ax = plt.subplots(figsize=(7.0, max(2.8, 0.55 * len(labels) + 1.7)))
    positions = list(range(len(labels)))

    base_row = mean_std_df[mean_std_df["Method"] == baseline_name]
    if not base_row.empty:
        base_mean = float(base_row[f"{metric} mean"].iloc[0])
        ax.axvline(base_mean, color=INK_SOFT, ls="--", lw=1.2, zorder=1)

    best_idx = max(positions, key=lambda i: means[i])
    for pos, mean, std, color in zip(positions, means, stds, colors):
        ax.errorbar(mean, pos, xerr=std, fmt="none", ecolor=color,
                    elinewidth=2.0, capsize=5, alpha=0.85, zorder=2)
        ax.plot(mean, pos, marker="o", markersize=9, color=color,
                markeredgecolor="#ffffff", markeredgewidth=1.6, zorder=3)

    for pos, mean, std in zip(positions, means, stds):
        ax.annotate(
            format_mean_std(mean, std if std else float("nan")),
            xy=(mean + std, pos), xytext=(10, 0), textcoords="offset points",
            va="center", fontsize=9, color=INK,
            fontweight="bold" if pos == best_idx else "normal",
        )

    low = min(m - s for m, s in zip(means, stds))
    high = max(m + s for m, s in zip(means, stds))
    span = max(high - low, 1e-6)
    ax.set_xlim(low - span * 0.25, high + span * 1.15)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)  # thứ tự từ trên xuống

    ax.grid(axis="x", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_title(f"{metric} theo strategy (mean ± std trên các seed)\n"
                 f"đường đứt = mức của {baseline_name}",
                 color=INK, pad=12, loc="left")
    ax.set_xlabel(metric, color=INK_SOFT)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _chart_delta(plt, per_seed_df, order, shorts, baseline_name, metric, out_path):
    """
    Δ so với baseline — thang diverging: xanh = tốt hơn, đỏ = tệ hơn, 0 là neutral.
    Chấm mờ là từng seed (paired), cột là trung bình.
    """
    if baseline_name is None:
        return None
    baseline = per_seed_df[per_seed_df["Method"] == baseline_name].set_index("Seed")
    labels, means, all_deltas = [], [], []
    for name in order:
        if name == baseline_name:
            continue
        group = per_seed_df[per_seed_df["Method"] == name].set_index("Seed")
        shared = [s for s in group.index if s in baseline.index]
        if not shared:
            continue
        deltas = [float(group.loc[s, metric]) - float(baseline.loc[s, metric]) for s in shared]
        labels.append(shorts.get(name, name))
        means.append(_mean(deltas))
        all_deltas.append(deltas)
    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(7.0, max(2.6, 0.62 * len(labels) + 1.7)))
    positions = list(range(len(labels)))
    colors = [POS if m >= 0 else NEG for m in means]

    # Lollipop (thân mảnh + đầu tròn) thay vì bar khối: vẫn neo ở 0 nên chiều
    # dài đọc đúng tỉ lệ, nhưng không phải mảng màu đặc chiếm nửa hình.
    ax.axvline(0, color=AXIS, lw=1.2, zorder=1)
    for pos, mean, color in zip(positions, means, colors):
        ax.hlines(pos, 0, mean, color=color, lw=2.5, zorder=2)
        ax.plot(mean, pos, marker="o", markersize=9, color=color,
                markeredgecolor="#ffffff", markeredgewidth=1.6, zorder=4)
    # Chấm từng seed đặt lệch xuống dưới thân để không đè lên đầu lollipop
    for pos, deltas in zip(positions, all_deltas):
        ax.scatter(deltas, [pos + 0.26] * len(deltas), s=24, color=MUTED, alpha=0.75,
                   zorder=3, edgecolors="#ffffff", linewidths=0.8)

    # Nhãn giá trị đặt trong lề phải riêng, sau mọi chấm seed → không bao giờ đè
    flat = [d for deltas in all_deltas for d in deltas] + means + [0.0]
    low, high = min(flat), max(flat)
    span = max(high - low, 1e-9)
    label_x = high + span * 0.06
    ax.set_xlim(low - span * 0.08, high + span * 0.45)
    for pos, mean in zip(positions, means):
        ax.annotate(f"{mean:+.4f}", xy=(label_x, pos), va="center", ha="left",
                    fontsize=9, color=INK)

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_ylim(len(labels) - 0.5, -0.6)
    ax.grid(axis="x", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_title(
        f"Δ {metric} so với {baseline_name}\n"
        "(đầu tròn xanh = trung bình, chấm xám = từng seed; >0 là tốt hơn)",
        color=INK, pad=12, loc="left",
    )
    ax.set_xlabel(f"Δ {metric}", color=INK_SOFT)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _chart_per_seed_paired(plt, per_seed_df, order, shorts, metric, out_path):
    """
    Paired dot plot: mỗi seed là một đường mảnh nối các strategy.

    Cho thấy đồng thời (a) biến thiên giữa các seed và (b) cấu trúc paired —
    strategy nào cải thiện nhất quán trên MỌI seed chứ không chỉ tốt trung bình.
    """
    seeds = sorted(per_seed_df["Seed"].dropna().unique().tolist())
    if not seeds or len(order) < 2:
        return None

    fig, ax = plt.subplots(figsize=(max(6.0, 1.15 * len(order) + 1.5), 4.0))
    positions = {name: i for i, name in enumerate(order)}

    for seed in seeds:
        group = per_seed_df[per_seed_df["Seed"] == seed]
        xs, ys = [], []
        for name in order:
            row = group[group["Method"] == name]
            if not row.empty:
                xs.append(positions[name])
                ys.append(float(row[metric].iloc[0]))
        if len(xs) > 1:
            ax.plot(xs, ys, color=MUTED, lw=1.0, alpha=0.55, zorder=2)
            ax.scatter(xs, ys, s=26, color=MUTED, alpha=0.7, zorder=3,
                       edgecolors="#ffffff", linewidths=0.8)

    means = []
    for name in order:
        values = [float(v) for v in per_seed_df[per_seed_df["Method"] == name][metric].tolist()]
        means.append(_mean(values) if values else float("nan"))
    ax.plot(list(positions.values()), means, color=POS, lw=2.0, zorder=5,
            marker="D", markersize=8, markerfacecolor=POS,
            markeredgecolor="#ffffff", markeredgewidth=1.6, label="mean")

    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels([shorts.get(n, n) for n in order],
                       rotation=0 if len(order) <= 6 else 25,
                       ha="center" if len(order) <= 6 else "right")
    _style_axes(ax, ylabel=metric,
                title=f"{metric} từng seed (đường xám = 1 seed, kim cương xanh = trung bình)")
    ax.legend(loc="best", fontsize=9)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def export_experiment_charts(mean_std_df, per_seed_df, order, shorts, baseline_name, charts_dir):
    """Vẽ toàn bộ chart tổng hợp của experiment vào <exp_dir>/charts/."""
    if not Config.MAKE_CHARTS:
        return []
    plt = _init_matplotlib()
    if plt is None:
        return []

    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def _try(fn, *args):
        try:
            path = fn(*args)
            if path:
                written.append(Path(path))
        except Exception as exc:  # chart lỗi không được làm hỏng cả run
            print(f"  ⚠ Bỏ qua 1 chart do lỗi: {type(exc).__name__}: {exc}")

    _try(_chart_topk_curve, plt, mean_std_df, baseline_name, PRIMARY_METRIC,
         charts_dir / "01_topk_curve_mAP50-95.png")
    _try(_chart_strategy_dots, plt, mean_std_df, baseline_name, PRIMARY_METRIC,
         charts_dir / "02_method_mAP50-95.png")
    _try(_chart_strategy_dots, plt, mean_std_df, baseline_name, "mAP@0.5",
         charts_dir / "03_method_mAP50.png")
    _try(_chart_delta, plt, per_seed_df, order, shorts, baseline_name, PRIMARY_METRIC,
         charts_dir / "04_delta_vs_baseline.png")
    _try(_chart_per_seed_paired, plt, per_seed_df, order, shorts, PRIMARY_METRIC,
         charts_dir / "05_per_seed_paired.png")

    if written:
        print(f"  ✓ {len(written)} chart tổng hợp → {charts_dir}")
    return written


# ==================== Dọn cây thư mục kết quả ====================

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".svg", ".pdf"}
SAMPLE_IMAGE_PREFIXES = ("train_batch", "val_batch")
# Chỉ xóa các file Ultralytics sinh mà nội dung đã nằm gọn trong Excel.
TIDY_DELETE_NAMES = {"results.csv", "args.yaml", "tune_results.csv"}


def tidy_run_dir(run_dir):
    """
    Rút gọn run dir của một seed xuống đúng những gì cần theo dõi:
    file Excel + thư mục charts/.

    - Ảnh/chart ở gốc run dir  → charts/training/
    - Ảnh sample train_batch*/val_batch* → xóa (nặng, chỉ để debug) trừ khi
      Config.KEEP_SAMPLE_IMAGES = True
    - results.csv / args.yaml  → xóa (đã có sheet PerEpoch + RunInfo trong Excel)
      trừ khi Config.KEEP_RESULTS_CSV = True
    - File lạ không nhận diện được → GIỮ NGUYÊN (không xóa mù)
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return
    charts_dir = run_dir / "charts"
    training_dir = charts_dir / "training"

    moved = removed = 0
    for path in sorted(run_dir.iterdir()):
        if path.is_dir():
            if path == charts_dir:
                continue
            # thư mục weights/ đã được xóa trước đó; các dir khác giữ nguyên
            continue
        name = path.name
        suffix = path.suffix.lower()

        if suffix in IMAGE_SUFFIXES:
            if name.startswith(SAMPLE_IMAGE_PREFIXES) and not Config.KEEP_SAMPLE_IMAGES:
                path.unlink()
                removed += 1
                continue
            training_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(training_dir / name))
            moved += 1
        elif name in TIDY_DELETE_NAMES and not Config.KEEP_RESULTS_CSV:
            path.unlink()
            removed += 1

    # Ảnh sample trong các thư mục eval per-strategy
    if not Config.KEEP_SAMPLE_IMAGES and charts_dir.exists():
        for path in charts_dir.rglob("*"):
            if path.is_file() and path.name.startswith(SAMPLE_IMAGE_PREFIXES):
                path.unlink()
                removed += 1

    # Bỏ thư mục rỗng còn sót để cây kết quả không có folder trống gây rối
    for path in sorted(run_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()

    print(f"  ✓ Dọn run dir: {moved} chart → charts/, xóa {removed} file tạm ({run_dir})")


def print_multi_seed_table(all_runs_results):
    """Bảng tổng kết in ra console sau khi chạy xong tất cả seed."""
    per_seed_df, _, _, order, shorts = _build_summary_frames(all_runs_results)
    if per_seed_df.empty:
        return

    seeds = sorted(per_seed_df["Seed"].dropna().unique().tolist())
    label_width = max([len(shorts.get(n, n)) for n in order] + [10])

    print("\n" + "=" * (label_width + 14 * len(seeds) + 24))
    print(f" MULTI-SEED SUMMARY — {PRIMARY_METRIC} trên split '{Config.EVAL_SPLIT}'")
    print("=" * (label_width + 14 * len(seeds) + 24))
    header = f"  {'Method':<{label_width}}" + "".join(f" | seed {s:<7}" for s in seeds)
    header += " | mean ± std"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name in order:
        group = per_seed_df[per_seed_df["Method"] == name].set_index("Seed")
        row = f"  {shorts.get(name, name):<{label_width}}"
        values = []
        for seed in seeds:
            if seed in group.index:
                value = float(group.loc[seed, PRIMARY_METRIC])
                values.append(value)
                row += f" | {value:<12.4f}"
            else:
                row += f" | {'—':<12}"
        row += f" | {format_mean_std(_mean(values), _std(values))}"
        print(row)
    print("=" * (label_width + 14 * len(seeds) + 24))
