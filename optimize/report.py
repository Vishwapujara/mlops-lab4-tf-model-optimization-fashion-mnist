"""Comparison table, deployment recommendation and plots."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REFERENCE = "baseline (float32)"
CONTROL = "finetune_control (float32)"
COLUMNS = ["variant", "technique", "accuracy", "accuracy_drop_pp", "worst_class_drop_pp", "worst_class",
           "size_kb", "gzip_kb", "gzip_reduction_x", "latency_ms", "speedup_x", "sparsity", "max_unique_values"]


def build_table(rows, class_names):
    """Add metrics relative to each variant's own uncompressed starting point.

    Accuracy: post-training quantization is compared with the trained baseline; every technique that
    fine-tunes is compared with the fine-tune control (same extra training, no compression), so a
    technique never looks better just because it trained longer.
    Size and speed: always relative to the float32 baseline file.
    """
    df = pd.DataFrame(rows)
    ref = df[df["variant"] == REFERENCE].iloc[0]
    control = df[df["variant"] == CONTROL].iloc[0]
    matched = [ref if t == "post-training quantization" else control for t in df["technique"]]
    ref_acc = np.array([m["accuracy"] for m in matched])
    ref_per_class = np.vstack([m["per_class"] for m in matched])
    class_drops = (ref_per_class - np.vstack(df["per_class"])) * 100
    df["accuracy_drop_pp"] = (ref_acc - df["accuracy"]) * 100
    df["worst_class_drop_pp"] = class_drops.max(axis=1)
    df["worst_class"] = [class_names[i] for i in class_drops.argmax(axis=1)]
    df["gzip_reduction_x"] = ref["gzip_kb"] / df["gzip_kb"]
    df["speedup_x"] = ref["latency_ms"] / df["latency_ms"]
    return df[COLUMNS]


def recommend(df, max_accuracy_drop_pp, max_class_drop_pp):
    """Smallest compressed model within the accuracy budget (overall AND per class); ties go to the faster one."""
    ok = df[(df["accuracy_drop_pp"] <= max_accuracy_drop_pp) & (df["worst_class_drop_pp"] <= max_class_drop_pp)]
    if ok.empty:
        return None
    return ok.sort_values(["gzip_kb", "latency_ms"]).iloc[0]


def to_markdown(df):
    fmt = {"accuracy": "{:.4f}", "accuracy_drop_pp": "{:+.2f}", "worst_class_drop_pp": "{:+.2f}",
           "size_kb": "{:.1f}", "gzip_kb": "{:.1f}", "gzip_reduction_x": "{:.1f}x", "latency_ms": "{:.3f}",
           "speedup_x": "{:.2f}x", "sparsity": "{:.0%}", "max_unique_values": "{:d}"}
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, row in df.iterrows():
        cells = [fmt[c].format(row[c]) if c in fmt else str(row[c]) for c in df.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def size_accuracy_plot(df, path, recommended=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    for technique, group in df.groupby("technique"):
        ax.scatter(group["gzip_kb"], group["accuracy"] * 100, label=technique, s=50)
    for _, row in df.iterrows():
        ax.annotate(row["variant"], (row["gzip_kb"], row["accuracy"] * 100), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    if recommended is not None:
        ax.scatter([recommended["gzip_kb"]], [recommended["accuracy"] * 100], s=250, facecolors="none",
                   edgecolors="black", linewidths=2, label="recommended")
    ax.set_xscale("log")
    ax.set(title="Compressed size vs test accuracy (Fashion-MNIST)", xlabel="gzipped model size (KB, log scale)",
           ylabel="test accuracy (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pruning_plot(df, path):
    pruned = df[df["technique"].str.startswith("pruning (")].copy()
    pruned["schedule"] = pruned["technique"].str.extract(r"pruning \((\w+)")[0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for schedule, group in pruned.groupby("schedule"):
        group = group.sort_values("sparsity")
        ax.plot(group["sparsity"] * 100, group["accuracy"] * 100, marker="o", label=f"{schedule} schedule")
    control = df[df["variant"] == CONTROL].iloc[0]
    ax.axhline(control["accuracy"] * 100, ls="--", color="grey", label="fine-tune control (no pruning)")
    ax.set(title="Accuracy vs pruning sparsity", xlabel="measured sparsity (%)", ylabel="test accuracy (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
