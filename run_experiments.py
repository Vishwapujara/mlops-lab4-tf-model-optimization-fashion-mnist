"""Train a Fashion-MNIST CNN, apply every compression technique, and compare them on equal terms.

  python run_experiments.py            # full run (~5 min on a laptop CPU)
  python run_experiments.py --quick    # small smoke test (~1 min)

Writes results/: results.csv, results.md, summary.json, plots/*.png and every model as models/*.tflite.
"""

import argparse
import json
import time
from pathlib import Path

import optimize  # noqa: F401  (sets TF_USE_LEGACY_KERAS before TensorFlow loads)
import tf_keras as keras
import yaml

from optimize import convert, data, evaluate, models, report
from optimize import techniques as T


def load_config(path, quick):
    cfg = yaml.safe_load(Path(path).read_text())
    if quick:
        q = cfg["quick"]
        cfg.update(epochs=q["epochs"], finetune_epochs=q["finetune_epochs"], latency_runs=q["latency_runs"])
        cfg["pruning"]["targets"] = q["targets"]
    cfg["train_limit"] = cfg["quick"]["train_limit"] if quick else None
    cfg["test_limit"] = cfg["quick"]["test_limit"] if quick else None
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--quick", action="store_true", help="tiny run for smoke tests")
    parser.add_argument("--out", default="results")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.quick)
    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    keras.utils.set_random_seed(cfg["seed"])
    started = time.time()

    (x_train, y_train), (x_test, y_test) = data.load(cfg["train_limit"], cfg["test_limit"])
    calibration = data.representative_dataset(x_train, seed=cfg["seed"])
    print(f"Fashion-MNIST: {len(x_train)} train / {len(x_test)} test images")
    rows = []

    def record(name, technique, model, modes):
        stats = T.weight_stats(model)
        for mode in modes:
            model_bytes = convert.to_tflite(model, mode, calibration)
            (out / "models" / f"{name}__{mode}.tflite").write_bytes(model_bytes)
            acc, per_class = evaluate.accuracy(y_test, evaluate.predict(model_bytes, x_test))
            row = {"variant": f"{name} ({mode})", "technique": technique, "accuracy": acc, "per_class": per_class,
                   **evaluate.sizes(model_bytes), "latency_ms": evaluate.latency_ms(model_bytes, x_test[0],
                                                                                    cfg["latency_runs"]), **stats}
            rows.append(row)
            print(f"  -> {row['variant']:<34} acc {acc:.4f} | {row['size_kb']:7.1f} KB "
                  f"(gzip {row['gzip_kb']:6.1f}) | {row['latency_ms']:.3f} ms | sparsity {stats['sparsity']:.0%}")

    print("\n[1/7] Baseline + post-training quantization")
    base = models.build_baseline()
    models.fit(base, x_train, y_train, cfg["epochs"], cfg["batch_size"])
    record("baseline", "post-training quantization", base, convert.MODES)

    print("\n[2/7] Fine-tune control (same extra training, no compression)")
    record("finetune_control", "control", T.finetune_control(base, x_train, y_train, cfg), ["float32"])

    print("\n[3/7] Quantization-aware training")
    record("qat", "quantization-aware training", T.quantization_aware(base, x_train, y_train, cfg), ["int8"])

    print("\n[4/7] Magnitude pruning sweep")
    chain_input = None
    for schedule in cfg["pruning"]["schedules"]:
        for target in cfg["pruning"]["targets"]:
            pruned = T.prune(base, x_train, y_train, cfg, target, schedule)
            record(f"prune_{schedule}_{int(target * 100)}", f"pruning ({schedule})", pruned, ["float32"])
            if schedule == "polynomial" and target == cfg["pruning"]["chain_sparsity"]:
                chain_input = pruned
    if chain_input is None:  # --quick only tries one target
        chain_input = T.prune(base, x_train, y_train, cfg, cfg["pruning"]["chain_sparsity"], "polynomial")

    print("\n[5/7] Structured 2:4 pruning")
    record("prune_2by4", "structured 2:4 pruning", T.prune_2_by_4(base, x_train, y_train, cfg), ["float32"])

    print("\n[6/7] Weight clustering")
    record(f"cluster_{cfg['clusters']}", "weight clustering",
           T.cluster(base, x_train, y_train, cfg, cfg["clusters"]), ["float32"])

    print("\n[7/7] Collaborative optimization (prune -> [cluster] -> QAT, int8)")
    chain = f"prune{int(cfg['pruning']['chain_sparsity'] * 100)}"
    record(f"{chain}_pqat", "collaborative", T.prune_preserving_qat(chain_input, x_train, y_train, cfg), ["int8"])
    record(f"{chain}_cluster{cfg['clusters']}_pcqat", "collaborative",
           T.prune_cluster_preserving_qat(chain_input, x_train, y_train, cfg, cfg["clusters"]), ["int8"])

    table = report.build_table(rows, data.CLASS_NAMES)
    budget = cfg["budget"]
    best = report.recommend(table, budget["max_accuracy_drop_pp"], budget["max_class_drop_pp"])
    table.to_csv(out / "results.csv", index=False)
    (out / "results.md").write_text(report.to_markdown(table) + "\n")
    report.size_accuracy_plot(table, out / "plots" / "size_vs_accuracy.png", best)
    report.pruning_plot(table, out / "plots" / "pruning_sparsity.png")

    ref = table[table["variant"] == report.REFERENCE].iloc[0]
    summary = {
        "baseline_float32": {"accuracy": round(ref["accuracy"], 4), "size_kb": round(ref["size_kb"], 1),
                             "gzip_kb": round(ref["gzip_kb"], 1), "latency_ms": round(ref["latency_ms"], 3)},
        "budget": budget,
        "recommended": None if best is None else {
            "variant": best["variant"], "accuracy": round(best["accuracy"], 4),
            "accuracy_drop_pp": round(best["accuracy_drop_pp"], 2),
            "worst_class": best["worst_class"], "worst_class_drop_pp": round(best["worst_class_drop_pp"], 2),
            "gzip_kb": round(best["gzip_kb"], 1), "gzip_reduction_x": round(best["gzip_reduction_x"], 1),
            "speedup_x": round(best["speedup_x"], 2)},
        "variants": len(table), "quick": args.quick, "runtime_s": round(time.time() - started),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + report.to_markdown(table))
    if best is None:
        print(f"\nNo model meets the budget ({budget}).")
    else:
        print(f"\nRECOMMENDED: {best['variant']} - {best['gzip_reduction_x']:.1f}x smaller (gzip), "
              f"{best['speedup_x']:.2f}x faster, accuracy {best['accuracy_drop_pp']:+.2f} pp, "
              f"worst class {best['worst_class']} {best['worst_class_drop_pp']:+.2f} pp")
    print(f"Results written to {out}/ in {summary['runtime_s']} s")
    return summary


if __name__ == "__main__":
    main()
