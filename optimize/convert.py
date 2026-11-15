"""Keras -> TF Lite conversion in four precisions."""

import tensorflow as tf

MODES = ["float32", "float16", "dynamic", "int8"]


def to_tflite(model, mode, representative_data=None):
    """Convert a Keras model to TF Lite bytes.

    float32 - no quantization (reference)
    float16 - weights stored as float16 (~2x smaller)
    dynamic - weights int8, activations float at run time (what the original lab called "quantized")
    int8    - full integer: weights, activations, inputs and outputs int8 (needs calibration data);
              the format edge TPUs and microcontrollers require
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if mode == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif mode == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif mode == "int8":
        if representative_data is None:
            raise ValueError("int8 conversion needs representative_data for calibration")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    elif mode != "float32":
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return converter.convert()
