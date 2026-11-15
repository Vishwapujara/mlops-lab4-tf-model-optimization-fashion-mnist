"""Fashion-MNIST loading and the representative dataset used for full-integer quantization."""

import numpy as np
import tf_keras as keras

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def load(train_limit=None, test_limit=None):
    """Return (x_train, y_train), (x_test, y_test) with pixels scaled to [0, 1] as float32."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = (x_train / 255.0).astype("float32")[:train_limit]
    x_test = (x_test / 255.0).astype("float32")[:test_limit]
    return (x_train, y_train[:train_limit]), (x_test, y_test[:test_limit])


def representative_dataset(x, n=200, seed=0):
    """Calibration samples the TF Lite converter uses to pick int8 ranges for activations."""
    idx = np.random.default_rng(seed).choice(len(x), size=min(n, len(x)), replace=False)

    def generator():
        for i in idx:
            yield [x[i:i + 1]]

    return generator
