"""Model optimization experiments: pruning, clustering and quantization of a Fashion-MNIST CNN."""

import os

# tensorflow-model-optimization only supports Keras 2, which ships as the tf_keras package.
# This must be set before TensorFlow is imported anywhere.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
