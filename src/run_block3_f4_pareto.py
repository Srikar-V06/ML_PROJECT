"""Step F4: Pareto frontier plots (Test PR AUC vs training time and inference latency)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

P3_ROOT = Path("artifacts/p3")


def pareto_frontier_indices(x_time, y_score):
    """Return indices of non-dominated points (minimize x_time, maximize y_score)."""
    order = np.argsort(x_time, kind="mergesort")
    frontier = []
    best_y = -np.inf
    for idx in order:
        if np.isnan(x_time[idx]) or np.isnan(y_score[idx]):
            continue
        if y_score[idx] > best_y:
            best_y = y_score[idx]
            frontier.append(int(idx))
    return frontier


def _plot_pareto(df, x_col, x_label, title, out_path):
    # Exclude constant dummy (0.20 PR AUC) and pure-zero-train avg_top3 if log-x needs positive floor
    sub = df.loc[df["model"] != "constant"].copy().reset_index(drop=True)
    x_raw = sub[x_col].to_numpy(dtype=float)
    x_plot = np.maximum(x_raw, 1e-2 if x_col == "train_seconds" else 1e-1)
    y = sub["pr_auc"].to_numpy(dtype=float)

    front_idx = pareto_frontier_indices(x_plot, y)
    front_set = set(front_idx)

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    markers = {"full_210k": "o", "full_210k_oof_168k": "D", "subsample_20k": "^", "subsample_50k": "s"}
    colors = {"P1": "#1f77b4", "P2": "#2ca02c", "P3": "#d62728"}

    for i, r in sub.iterrows():
        mk = markers.get(r["subsample_label"], "v")
        c = colors.get(r["block"], "#7f7f7f")
        edge = "black" if i in front_set else "white"
        size = 110 if i in front_set else 60
        ax.scatter(x_plot[i], y[i], s=size, marker=mk, color=c, edgecolor=edge, linewidth=1.4, alpha=0.9, zorder=4)

    # Connect Pareto frontier
    fx = x_plot[front_idx]
    fy = y[front_idx]
    ax.step(fx, fy, where="post", color="black", linestyle="--", linewidth=1.6, alpha=0.8, label="Pareto frontier", zorder=3)

    # Label Pareto models + key reference models
    label_set = front_set | set(sub.index[sub["model"].isin({
        "stack_lr", "avg_top3", "catboost", "xgb_hist", "lgbm_native_flags", "lgbm_monotone",
        "hgb", "gb_classic", "adaboost", "logreg_baseline", "woe_scorecard", "random_forest",
    })])
    for i in sorted(label_set):
        name = sub.loc[i, "model"]
        tag = f" [{sub.loc[i, 'subsample_label']}]" if "subsample" in str(sub.loc[i, "subsample_label"]) else ""
        dy = 5 if (i % 2 == 0) else -11
        ax.annotate(
            f"{name}{tag}",
            (x_plot[i], y[i]),
            xytext=(6, dy),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold" if i in front_set else "normal",
            zorder=5,
        )

    ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Test PR AUC (Average Precision)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, which="both", alpha=0.25)

    # Legend for blocks & markers
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="black", linestyle="--", lw=1.6, label="Pareto frontier"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=8, label="P1 (Baselines)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markersize=8, label="P2 (Classical / PCA)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markersize=8, label="P3 (Boosters / Ensembles)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markersize=8, label="Subsample-fitted (20k/50k)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return sub.iloc[front_idx]["model"].tolist()


def run():
    bm = pd.read_csv(P3_ROOT / "tables" / "benchmark.csv")
    train_front = _plot_pareto(
        bm,
        "train_seconds",
        "Training Time (seconds, log scale; avg_top3 plotted at 0.01s floor)",
        "Analysis C Pareto Frontier: Test PR AUC vs Training Time",
        P3_ROOT / "plots" / "pareto_train_time.png",
    )
    infer_front = _plot_pareto(
        bm,
        "infer_ms_per_1000",
        "Inference Latency per 1,000 Rows (ms, log scale, 1 thread)",
        "Analysis C Pareto Frontier: Test PR AUC vs Real-Time Inference Latency",
        P3_ROOT / "plots" / "pareto_infer_time.png",
    )
    rows = (
        [{"frontier": "train_time", "model": m} for m in train_front]
        + [{"frontier": "infer_time", "model": m} for m in infer_front]
    )
    pd.DataFrame(rows).to_csv(P3_ROOT / "tables" / "pareto_frontier.csv", index=False)
    print(f"[F4] Train-time Pareto frontier: {train_front}", flush=True)
    print(f"[F4] Infer-time Pareto frontier: {infer_front}", flush=True)


if __name__ == "__main__":
    run()
