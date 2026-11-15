import numpy as np
import pytest

import optimize  # noqa: F401  (sets TF_USE_LEGACY_KERAS before TensorFlow loads)
from optimize import models


@pytest.fixture(scope="session")
def toy_data():
    """Learnable 28x28 images: each class lights up its own 6x6 patch on top of noise."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, 800)
    x = rng.random((800, 28, 28)).astype("float32") * 0.3
    for i, c in enumerate(y):
        r, col = divmod(int(c), 5)
        x[i, 2 + r * 12: 8 + r * 12, 1 + col * 5: 6 + col * 5] += 0.7
    return x, y


@pytest.fixture(scope="session")
def cfg():
    return {"batch_size": 64, "finetune_epochs": 1, "finetune_lr": 1e-3}


@pytest.fixture(scope="session")
def trained(toy_data):
    x, y = toy_data
    model = models.build_baseline()
    models.fit(model, x, y, epochs=3, batch_size=64)
    return model
