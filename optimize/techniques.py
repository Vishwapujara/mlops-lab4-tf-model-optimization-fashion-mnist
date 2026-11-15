"""Compression techniques from the TensorFlow Model Optimization Toolkit.

Every technique starts from a copy of the same trained baseline and is fine-tuned with the same
number of epochs and learning rate, so differences come from the technique, not from extra training.
"""

import numpy as np
import tensorflow_model_optimization as tfmot

from optimize.models import clone, fit

sparsity = tfmot.sparsity.keras
clustering = tfmot.clustering.keras
quantization = tfmot.quantization.keras
combine = tfmot.experimental.combine

# A Conv2D/Dense layer's trainable weight matrix; wrappers (prune/cluster/quantize) keep it on .layer.
_KERNEL_LAYERS = ("Conv2D", "Dense")


def _steps(n_samples, cfg):
    return int(np.ceil(n_samples * 0.9 / cfg["batch_size"])) * cfg["finetune_epochs"]


def _finetune(model, x, y, cfg, callbacks=()):
    fit(model, x, y, cfg["finetune_epochs"], cfg["batch_size"], callbacks, cfg["finetune_lr"])
    return model


def finetune_control(base, x, y, cfg):
    """Same extra training as every technique but no compression - the fair reference point."""
    return _finetune(clone(base), x, y, cfg)


def quantization_aware(base, x, y, cfg):
    """Insert fake-quant nodes and fine-tune, so the weights adapt to int8 precision before conversion."""
    return _finetune(quantization.quantize_model(clone(base)), x, y, cfg)


def prune(base, x, y, cfg, target, schedule):
    """Magnitude pruning to `target` sparsity.

    polynomial - sparsity ramps from 0 to target over the first 75% of fine-tuning, then the
                 network recovers at the final sparsity (gradual, usually more accurate)
    constant   - target sparsity applied from the first step
    """
    total = _steps(len(x), cfg)
    end = int(total * 0.75)
    # Update the masks ~10 times during the ramp (a fixed 50 would never fire on short runs).
    frequency = max(1, min(50, end // 10))
    if schedule == "polynomial":
        sched = sparsity.PolynomialDecay(initial_sparsity=0.0, final_sparsity=target,
                                         begin_step=0, end_step=end, frequency=frequency)
    elif schedule == "constant":
        sched = sparsity.ConstantSparsity(target, begin_step=0, frequency=frequency)
    else:
        raise ValueError(f"schedule must be 'polynomial' or 'constant', got {schedule!r}")
    model = sparsity.prune_low_magnitude(clone(base), pruning_schedule=sched)
    _finetune(model, x, y, cfg, [sparsity.UpdatePruningStep()])
    return sparsity.strip_pruning(model)


def prune_2_by_4(base, x, y, cfg):
    """Structured 2:4 sparsity - in every block of 4 weights, 2 are zero (the pattern sparse GPU kernels accelerate)."""
    model = sparsity.prune_low_magnitude(clone(base), sparsity_m_by_n=(2, 4))
    _finetune(model, x, y, cfg, [sparsity.UpdatePruningStep()])
    return sparsity.strip_pruning(model)


def cluster(model, x, y, cfg, n_clusters, preserve_sparsity=False):
    """Weight clustering: each layer's weights are snapped to `n_clusters` shared values (k-means++ init)."""
    clustered = clustering.cluster_weights(
        clone(model), number_of_clusters=n_clusters,
        cluster_centroids_init=clustering.CentroidInitialization.KMEANS_PLUS_PLUS,
        preserve_sparsity=preserve_sparsity)
    _finetune(clustered, x, y, cfg)
    return clustering.strip_clustering(clustered)


def prune_preserving_qat(pruned, x, y, cfg):
    """PQAT: quantization-aware training that keeps the pruned zeros at zero."""
    annotated = quantization.quantize_annotate_model(clone(pruned))
    model = quantization.quantize_apply(annotated, combine.Default8BitPrunePreserveQuantizeScheme())
    return _finetune(model, x, y, cfg)


def prune_cluster_preserving_qat(pruned, x, y, cfg, n_clusters):
    """PCQAT: sparsity-preserving clustering, then QAT that keeps both the zeros and the clusters."""
    clustered = cluster(pruned, x, y, cfg, n_clusters, preserve_sparsity=True)
    annotated = quantization.quantize_annotate_model(clustered)
    model = quantization.quantize_apply(
        annotated, combine.Default8BitClusterPreserveQuantizeScheme(preserve_sparsity=True))
    return _finetune(model, x, y, cfg)


def weight_stats(model):
    """Share of zero weights and the max number of distinct values in any Conv/Dense kernel."""
    kernels = []
    for layer in model.layers:
        inner = getattr(layer, "layer", layer)
        if type(inner).__name__ in _KERNEL_LAYERS:
            # Read the stored variable: inside a QAT wrapper, .kernel is the fake-quantized tensor.
            variable = next(w for w in layer.weights if w.name.split("/")[-1].startswith("kernel"))
            kernels.append(variable.numpy())
    total = sum(k.size for k in kernels)
    zeros = sum(int((k == 0).sum()) for k in kernels)
    return {"weights": int(total), "sparsity": zeros / total,
            "max_unique_values": int(max(len(np.unique(k)) for k in kernels))}
