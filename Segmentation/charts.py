"""
Chart cho báo cáo thí nghiệm Top-K checkpoint averaging.

Toàn bộ figure được vẽ bằng matplotlib với backend "Agg" (không cần display,
chạy được trên server). Mỗi hàm nhận DataFrame đã gom sẵn và ghi ra một file
PNG; nếu thiếu dữ liệu thì bỏ qua và trả về None thay vì raise, để lỗi vẽ
chart không bao giờ làm hỏng một run training đã tốn nhiều giờ.

Quy ước tên strategy (xem evaluate.py):
    "Baseline (best.pt)"        -> baseline, không có K
    "Top-1 (best raw ckpt)"       -> K = 1, không average
    "CWA (Top-<K> avg)"    -> K = 2, 3, ...
``strategy_k()`` parse K từ tên để dựng đường cong metric theo K.
"""
import re
from pathlib import Path

# Palette an toàn cho người mù màu (Okabe–Ito), dùng thống nhất mọi figure.
BASELINE_COLOR = "#D55E00"   # cam đỏ  — Baseline / baseline
AVERAGE_COLOR = "#0072B2"    # xanh dương — CWA (averaging)
SINGLE_COLOR = "#009E73"     # xanh lá   — Top-1 raw (không average)
NEUTRAL_COLOR = "#666666"
SEED_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]

DPI = 200


def strategy_k(name):
    """Parse K từ tên strategy; trả None nếu strategy không thuộc họ Top-K."""
    match = re.search(r"Top-(\d+)", str(name))
    return int(match.group(1)) if match else None


def strategy_color(name):
    k = strategy_k(name)
    if k is None:
        return BASELINE_COLOR
    return SINGLE_COLOR if k == 1 else AVERAGE_COLOR


def _pyplot():
    """Import matplotlib với backend Agg; trả None nếu môi trường không có."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - phụ thuộc môi trường
        print(f"  ⚠ Không vẽ được chart (matplotlib lỗi: {exc})")
        return None

    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    })
    return plt


def _short(name):
    """Rút gọn tên strategy cho nhãn trục x."""
    k = strategy_k(name)
    if k is None:
        return "best.pt\n(baseline)"
    if k == 1:
        return "Top-1\n(raw)"
    return f"Top-{k}\navg"


def _finish(fig, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return out_path


def _ordered_strategies(summary_df):
    """Sắp xếp strategy: baseline trước, rồi Top-K tăng dần."""
    names = list(dict.fromkeys(summary_df["Method"]))
    return sorted(names, key=lambda n: (strategy_k(n) is not None, strategy_k(n) or 0))


# ==================== Chart tổng hợp nhiều seed ====================

def plot_metric_by_strategy(summary_df, metric, out_path, title=None, ylabel=None):
    """Bar chart mean ± std (qua các seed) của một metric cho từng strategy."""
    plt = _pyplot()
    if plt is None or metric not in summary_df.columns or summary_df.empty:
        return None

    strategies = _ordered_strategies(summary_df)
    means, stds, counts = [], [], []
    for name in strategies:
        values = summary_df.loc[summary_df["Method"] == name, metric].astype(float)
        means.append(values.mean())
        stds.append(values.std(ddof=1) if len(values) > 1 else 0.0)
        counts.append(len(values))

    fig, ax = plt.subplots(figsize=(1.25 * len(strategies) + 2.2, 3.8))
    positions = range(len(strategies))
    colors = [strategy_color(n) for n in strategies]
    ax.bar(positions, means, width=0.62, color=colors, zorder=3)
    ax.errorbar(
        positions, means, yerr=stds, fmt="none",
        ecolor="#222222", elinewidth=1.1, capsize=4, capthick=1.1, zorder=4,
    )

    baseline = means[0] if means else None
    if baseline is not None:
        ax.axhline(baseline, color=BASELINE_COLOR, linestyle="--", linewidth=1, alpha=0.7, zorder=2)

    for pos, mean, std in zip(positions, means, stds):
        ax.annotate(
            f"{mean:.4f}\n±{std:.4f}",
            (pos, mean + std),
            textcoords="offset points", xytext=(0, 4),
            ha="center", va="bottom", fontsize=7.5, color="#222222",
        )

    lo = min(m - s for m, s in zip(means, stds))
    hi = max(m + s for m, s in zip(means, stds))
    pad = max((hi - lo) * 0.45, hi * 0.02, 1e-4)
    ax.set_ylim(max(0.0, lo - pad), hi + pad * 1.6)
    ax.set_xticks(list(positions))
    ax.set_xticklabels([_short(n) for n in strategies])
    ax.set_ylabel(ylabel or metric)
    n_seeds = max(counts) if counts else 0
    ax.set_title(title or f"{metric} theo strategy (mean ± std, n={n_seeds} seeds)")
    ax.set_axisbelow(True)
    return _finish(fig, out_path)


def plot_topk_curve(summary_df, metric, out_path, title=None):
    """
    Đường cong metric theo K (mean ± std qua các seed), kèm đường ngang baseline
    Baseline. Đây là figure chính cho câu hỏi "average bao nhiêu checkpoint
    là tốt nhất".
    """
    plt = _pyplot()
    if plt is None or metric not in summary_df.columns or summary_df.empty:
        return None

    points = []
    for name in _ordered_strategies(summary_df):
        k = strategy_k(name)
        if k is None:
            continue
        values = summary_df.loc[summary_df["Method"] == name, metric].astype(float)
        points.append((k, values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0))
    if len(points) < 2:
        return None
    points.sort()
    ks = [p[0] for p in points]
    means = [p[1] for p in points]
    stds = [p[2] for p in points]

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.fill_between(
        ks,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        color=AVERAGE_COLOR, alpha=0.15, linewidth=0, zorder=2,
    )
    ax.plot(ks, means, marker="o", markersize=5, color=AVERAGE_COLOR,
            linewidth=1.8, zorder=4, label="Top-K average")

    baseline_rows = summary_df[summary_df["Method"].map(strategy_k).isna()]
    if not baseline_rows.empty:
        baseline = float(baseline_rows[metric].astype(float).mean())
        ax.axhline(baseline, color=BASELINE_COLOR, linestyle="--", linewidth=1.4,
                   zorder=3, label="Baseline (best.pt)")

    best_idx = max(range(len(means)), key=lambda i: means[i])
    ax.annotate(
        f"K={ks[best_idx]}\n{means[best_idx]:.4f}",
        (ks[best_idx], means[best_idx]),
        textcoords="offset points", xytext=(0, 12),
        ha="center", fontsize=8, color=AVERAGE_COLOR, fontweight="bold",
    )

    ax.set_xticks(ks)
    ax.set_xlabel("K (số checkpoint được average)")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} theo K (mean ± std qua các seed)")
    ax.legend(loc="best")
    ax.set_axisbelow(True)
    return _finish(fig, out_path)


def plot_per_seed(summary_df, metric, out_path, title=None):
    """Mỗi seed một đường — cho thấy strategy thắng có nhất quán giữa các seed."""
    plt = _pyplot()
    if plt is None or metric not in summary_df.columns or summary_df.empty:
        return None

    strategies = _ordered_strategies(summary_df)
    seeds = list(dict.fromkeys(summary_df["Seed"]))
    if len(strategies) < 2:
        return None

    fig, ax = plt.subplots(figsize=(1.15 * len(strategies) + 3.0, 3.9))
    positions = list(range(len(strategies)))
    for i, seed in enumerate(seeds):
        rows = summary_df[summary_df["Seed"] == seed].set_index("Method")
        values = [float(rows[metric].get(name, float("nan"))) for name in strategies]
        ax.plot(
            positions, values, marker="o", markersize=4, linewidth=1.2,
            color=SEED_COLORS[i % len(SEED_COLORS)], alpha=0.9, label=f"seed {seed}",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([_short(n) for n in strategies])
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} từng seed theo strategy")
    ax.legend(loc="best", ncol=min(len(seeds), 3), fontsize=8)
    ax.set_axisbelow(True)
    return _finish(fig, out_path)


def plot_delta_vs_baseline(summary_df, metric, baseline_name, out_path, title=None):
    """
    Chênh lệch GHÉP CẶP theo seed so với baseline (Δ = strategy − baseline trên
    cùng seed), vẽ mean ± std. Paired delta là cách đọc đúng khi variance giữa
    các seed lớn hơn khoảng cách giữa các strategy.
    """
    plt = _pyplot()
    if plt is None or metric not in summary_df.columns or summary_df.empty:
        return None
    if baseline_name not in set(summary_df["Method"]):
        return None

    base = summary_df[summary_df["Method"] == baseline_name].set_index("Seed")[metric].astype(float)
    strategies = [n for n in _ordered_strategies(summary_df) if n != baseline_name]
    if not strategies:
        return None

    means, stds, wins = [], [], []
    for name in strategies:
        rows = summary_df[summary_df["Method"] == name].set_index("Seed")[metric].astype(float)
        deltas = (rows - base).dropna()
        means.append(float(deltas.mean()) if len(deltas) else 0.0)
        stds.append(float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0)
        wins.append(int((deltas > 0).sum()))

    n_seeds = len(base)
    fig, ax = plt.subplots(figsize=(1.25 * len(strategies) + 2.4, 3.9))
    positions = list(range(len(strategies)))
    colors = [AVERAGE_COLOR if m >= 0 else BASELINE_COLOR for m in means]
    ax.bar(positions, means, width=0.62, color=colors, zorder=3)
    ax.errorbar(positions, means, yerr=stds, fmt="none", ecolor="#222222",
                elinewidth=1.1, capsize=4, capthick=1.1, zorder=4)
    ax.axhline(0, color=NEUTRAL_COLOR, linewidth=1, zorder=2)

    for pos, mean, std, win in zip(positions, means, stds, wins):
        offset = 5 if mean >= 0 else -5
        ax.annotate(
            f"{mean:+.4f}\n{win}/{n_seeds} seeds tốt hơn",
            (pos, mean + (std if mean >= 0 else -std)),
            textcoords="offset points", xytext=(0, offset),
            ha="center", va="bottom" if mean >= 0 else "top",
            fontsize=7.5, color="#222222",
        )

    span = max(abs(m) + s for m, s in zip(means, stds)) or 1e-4
    ax.set_ylim(-span * 2.0, span * 2.0)
    ax.set_xticks(positions)
    ax.set_xticklabels([_short(n) for n in strategies])
    ax.set_ylabel(f"Δ {metric} vs baseline")
    ax.set_title(title or f"Δ {metric} so với {baseline_name} (paired theo seed)")
    ax.set_axisbelow(True)
    return _finish(fig, out_path)


def plot_per_class_delta(per_class_df, baseline_name, target_name, out_path,
                         metric="AP@0.5:0.95", top_n=None):
    """Δ AP từng class giữa strategy tốt nhất và baseline (mean qua các seed)."""
    plt = _pyplot()
    if plt is None or per_class_df is None or per_class_df.empty:
        return None
    names = set(per_class_df["Method"])
    if baseline_name not in names or target_name not in names or metric not in per_class_df.columns:
        return None

    base = per_class_df[per_class_df["Method"] == baseline_name].groupby("Class")[metric].mean()
    target = per_class_df[per_class_df["Method"] == target_name].groupby("Class")[metric].mean()
    deltas = (target - base).dropna().sort_values()
    if deltas.empty:
        return None
    if top_n and len(deltas) > top_n:
        deltas = deltas.iloc[list(range(top_n // 2)) + list(range(-(top_n - top_n // 2), 0))]

    fig, ax = plt.subplots(figsize=(6.2, max(3.0, 0.26 * len(deltas) + 1.4)))
    colors = [AVERAGE_COLOR if v >= 0 else BASELINE_COLOR for v in deltas.values]
    ax.barh(range(len(deltas)), deltas.values, color=colors, height=0.7, zorder=3)
    ax.axvline(0, color=NEUTRAL_COLOR, linewidth=1, zorder=2)
    ax.set_yticks(range(len(deltas)))
    ax.set_yticklabels(deltas.index, fontsize=8)
    ax.set_xlabel(f"Δ {metric}")
    ax.set_title(f"Δ {metric} theo class: {target_name} − {baseline_name}", fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    return _finish(fig, out_path)


# ==================== Chart theo từng seed ====================

def plot_checkpoint_selection(per_epoch_df, ranked, out_path, seed=None):
    """
    Fitness theo epoch + đánh dấu các checkpoint được chọn vào Top-K.

    Đây là figure minh họa trực tiếp phần "criteria-based selection" của paper:
    thấy ngay Top-K rơi vào vùng epoch nào và cách nhau bao xa.

    Args:
        per_epoch_df: DataFrame từ results.csv của Ultralytics.
        ranked: list[(path, fitness, epoch)] đã sort giảm dần theo fitness.
    """
    plt = _pyplot()
    if plt is None or per_epoch_df is None or per_epoch_df.empty:
        return None

    columns = ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/mAP50(M)", "metrics/mAP50-95(M)")
    if not all(column in per_epoch_df.columns for column in columns) or "epoch" not in per_epoch_df.columns:
        return None

    epochs = per_epoch_df["epoch"].astype(int).tolist()
    fitness = (
        0.1 * per_epoch_df["metrics/mAP50(B)"].astype(float)
        + 0.9 * per_epoch_df["metrics/mAP50-95(B)"].astype(float)
        + 0.1 * per_epoch_df["metrics/mAP50(M)"].astype(float)
        + 0.9 * per_epoch_df["metrics/mAP50-95(M)"].astype(float)
    ).tolist()

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(epochs, fitness, color=NEUTRAL_COLOR, linewidth=1.3, zorder=3, label="fitness (val)")

    if ranked:
        # epoch trong ranked là 0-based (tên file epochN.pt), results.csv 1-based.
        selected = {int(epoch) + 1: float(value) for _, value, epoch in ranked}
        xs = [e for e in epochs if e in selected]
        ys = [selected[e] for e in xs]
        if xs:
            ax.scatter(xs, ys, s=52, color=AVERAGE_COLOR, zorder=5,
                       label=f"Top-{len(ranked)} được giữ", edgecolors="white", linewidths=0.8)
            best_epoch = max(selected, key=lambda e: selected[e])
            ax.scatter([best_epoch], [selected[best_epoch]], s=95, facecolors="none",
                       edgecolors=BASELINE_COLOR, linewidths=1.6, zorder=6, label="rank 1 (best.pt)")

    ax.set_xlabel("epoch")
    ax.set_ylabel("fitness = 0.1·mAP50 + 0.9·mAP50-95  (box + mask)")
    ax.set_title(f"Checkpoint selection criterion{f' — seed {seed}' if seed is not None else ''}")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_axisbelow(True)
    return _finish(fig, out_path)


# ==================== Orchestration ====================

def build_summary_charts(summary_df, per_class_df, charts_dir, baseline_name):
    """Sinh toàn bộ chart tổng hợp; trả list path đã ghi (bỏ qua chart thiếu data)."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    written = []

    plans = [
        (plot_metric_by_strategy, (summary_df, "mAP@0.5:0.95", charts_dir / "01_mAP50-95_by_strategy.png")),
        (plot_metric_by_strategy, (summary_df, "mAP@0.5", charts_dir / "02_mAP50_by_strategy.png")),
        (plot_topk_curve, (summary_df, "mAP@0.5:0.95", charts_dir / "03_topk_curve_mAP50-95.png")),
        (plot_per_seed, (summary_df, "mAP@0.5:0.95", charts_dir / "04_per_seed_mAP50-95.png")),
        (plot_delta_vs_baseline,
         (summary_df, "mAP@0.5:0.95", baseline_name, charts_dir / "05_delta_vs_baseline.png")),
    ]
    for func, args in plans:
        try:
            path = func(*args)
        except Exception as exc:  # chart lỗi không được làm hỏng run
            print(f"  ⚠ Bỏ qua chart {func.__name__}: {exc}")
            continue
        if path:
            written.append(path)

    # Δ AP theo class giữa baseline và strategy averaging tốt nhất
    try:
        averaging = [n for n in set(summary_df["Method"]) if (strategy_k(n) or 0) > 1]
        if averaging and not per_class_df.empty:
            best = max(
                averaging,
                key=lambda n: float(summary_df.loc[summary_df["Method"] == n, "mAP@0.5:0.95"].mean()),
            )
            path = plot_per_class_delta(
                per_class_df, baseline_name, best, charts_dir / "06_per_class_delta.png"
            )
            if path:
                written.append(path)
    except Exception as exc:
        print(f"  ⚠ Bỏ qua chart per-class delta: {exc}")

    return written
