import json

import pytest

import run_experiments
from optimize import report

NAMES = [f"c{i}" for i in range(10)]


def row(variant, technique, acc, gzip_kb, latency=1.0, per_class=None):
    return {"variant": variant, "technique": technique, "accuracy": acc, "per_class": per_class or [acc] * 10,
            "size_kb": gzip_kb * 1.1, "gzip_kb": gzip_kb, "latency_ms": latency,
            "sparsity": 0.0, "max_unique_values": 100, "weights": 1}


@pytest.fixture
def table():
    rows = [
        row("baseline (float32)", "post-training quantization", 0.90, 800),
        row("baseline (int8)", "post-training quantization", 0.895, 200, latency=0.5),
        row("finetune_control (float32)", "control", 0.92, 800),
        # Compared with the control (0.92), not the baseline: a 1.5 pp drop, over budget.
        row("prune_90 (float32)", "pruning (polynomial)", 0.905, 100),
        # Fine overall, but one class collapses.
        row("cluster (float32)", "weight clustering", 0.915, 50, per_class=[0.92] * 9 + [0.80]),
    ]
    return report.build_table(rows, NAMES)


def test_drops_use_the_matched_reference(table):
    drop = dict(zip(table["variant"], table["accuracy_drop_pp"].round(2)))
    assert drop["baseline (int8)"] == 0.5          # vs baseline
    assert drop["prune_90 (float32)"] == 1.5       # vs fine-tune control
    assert table.loc[table["variant"] == "baseline (int8)", "gzip_reduction_x"].item() == 4


def test_recommendation_respects_both_budgets(table):
    best = report.recommend(table, max_accuracy_drop_pp=1.0, max_class_drop_pp=3.0)
    assert best["variant"] == "baseline (int8)"     # smaller models fail the overall or per-class budget
    assert report.recommend(table, 1.0, 20.0)["variant"] == "cluster (float32)"
    assert report.recommend(table, -5, -5) is None


def test_markdown_has_every_row(table):
    assert report.to_markdown(table).count("\n") == len(table) + 1


@pytest.mark.slow
def test_quick_run_end_to_end(tmp_path):
    """Full pipeline on a small slice of Fashion-MNIST (downloads the dataset on first run)."""
    summary = run_experiments.main(["--quick", "--out", str(tmp_path)])
    assert summary["variants"] == 12  # 4 PTQ + control + QAT + 2 pruning + 2:4 + cluster + PQAT + PCQAT
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["baseline_float32"]["size_kb"] > 800
    assert len(list((tmp_path / "models").glob("*.tflite"))) == 12
    assert (tmp_path / "plots" / "size_vs_accuracy.png").exists()
    assert (tmp_path / "plots" / "pruning_sparsity.png").exists()
