import numpy as np
import pytest
from ai_edge_litert.interpreter import Interpreter

from optimize import convert, data, evaluate


@pytest.fixture(scope="module")
def tflite_models(trained, toy_data):
    calibration = data.representative_dataset(toy_data[0], n=50)
    return {mode: convert.to_tflite(trained, mode, calibration) for mode in convert.MODES}


def test_quantized_files_are_smaller(tflite_models):
    size = {m: len(b) for m, b in tflite_models.items()}
    assert size["float16"] < 0.6 * size["float32"]
    assert size["dynamic"] < 0.3 * size["float32"]
    assert size["int8"] < 0.3 * size["float32"]


def test_int8_model_takes_int8_input(tflite_models):
    interpreter = Interpreter(model_content=tflite_models["int8"])
    assert interpreter.get_input_details()[0]["dtype"] == np.int8
    assert interpreter.get_output_details()[0]["dtype"] == np.int8


def test_every_precision_agrees_with_keras(trained, toy_data, tflite_models):
    x, _ = toy_data
    keras_pred = trained.predict(x, verbose=0).argmax(axis=1)
    for mode, model_bytes in tflite_models.items():
        agreement = (evaluate.predict(model_bytes, x, batch_size=300) == keras_pred).mean()
        assert agreement > 0.97, mode


def test_accuracy_sizes_latency(toy_data, tflite_models):
    x, y = toy_data
    acc, per_class = evaluate.accuracy(y, evaluate.predict(tflite_models["float32"], x))
    assert 0.9 < acc <= 1 and len(per_class) == 10
    s = evaluate.sizes(tflite_models["float32"])
    assert s["gzip_kb"] < s["size_kb"]
    assert evaluate.latency_ms(tflite_models["int8"], x[0], runs=5, warmup=1) > 0


def test_bad_conversion_settings(trained):
    with pytest.raises(ValueError, match="representative_data"):
        convert.to_tflite(trained, "int8")
    with pytest.raises(ValueError, match="mode"):
        convert.to_tflite(trained, "int4")
