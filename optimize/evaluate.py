"""Measure every TF Lite model the same way: accuracy, per-class accuracy, size and latency."""

import gzip
import time

import numpy as np
from ai_edge_litert.interpreter import Interpreter


def sizes(model_bytes):
    """Raw file size and gzip size (pruning and clustering only shrink the compressed file)."""
    return {"size_kb": len(model_bytes) / 1024, "gzip_kb": len(gzip.compress(model_bytes, 9)) / 1024}


def _quantize_input(batch, detail):
    if detail["dtype"] == np.int8:
        scale, zero_point = detail["quantization"]
        return np.clip(np.round(batch / scale + zero_point), -128, 127).astype(np.int8)
    return batch.astype(detail["dtype"])


def predict(model_bytes, x, batch_size=500):
    """Class predictions from a TF Lite model (handles int8 input quantization)."""
    interpreter = Interpreter(model_content=model_bytes)
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    preds, current = [], None
    for start in range(0, len(x), batch_size):
        batch = x[start:start + batch_size]
        if len(batch) != current:
            interpreter.resize_tensor_input(inp["index"], [len(batch), *x.shape[1:]])
            interpreter.allocate_tensors()
            current = len(batch)
        interpreter.set_tensor(inp["index"], _quantize_input(batch, inp))
        interpreter.invoke()
        # argmax is unaffected by output dequantization, so int8 outputs can be used directly.
        preds.append(interpreter.get_tensor(out["index"]).argmax(axis=1))
    return np.concatenate(preds)


def accuracy(y_true, y_pred, n_classes=10):
    per_class = [float((y_pred[y_true == c] == c).mean()) for c in range(n_classes)]
    return float((y_true == y_pred).mean()), per_class


def latency_ms(model_bytes, sample, runs=200, warmup=20):
    """Median single-image latency on one CPU thread (like a small edge device)."""
    interpreter = Interpreter(model_content=model_bytes, num_threads=1)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    data = _quantize_input(sample[None, ...], inp)
    times = []
    for i in range(warmup + runs):
        start = time.perf_counter()
        interpreter.set_tensor(inp["index"], data)
        interpreter.invoke()
        if i >= warmup:
            times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))
