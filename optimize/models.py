"""Baseline CNN and the shared training loop."""

import tf_keras as keras


def build_baseline():
    """Two conv blocks + dense head (~225k weights), big enough for compression to be visible."""
    return keras.Sequential([
        keras.layers.InputLayer(input_shape=(28, 28)),
        keras.layers.Reshape((28, 28, 1)),
        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ], name="fashion_cnn")


def compile_model(model, learning_rate=1e-3):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def fit(model, x, y, epochs, batch_size=128, callbacks=(), learning_rate=1e-3):
    """Compile and train with a fixed 10% validation split, so every technique gets the same budget."""
    compile_model(model, learning_rate)
    history = model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                        callbacks=list(callbacks), verbose=2)
    return history.history


def clone(model):
    """Fresh copy with the same weights, so each technique starts from the same trained baseline."""
    copy = keras.models.clone_model(model)
    copy.set_weights(model.get_weights())
    return copy
