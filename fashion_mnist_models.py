#!/usr/bin/env python3
"""Generate small Fashion-MNIST QKeras model variants for CI demos.

Variants:
  fashion_dense_8_bit - dense-heavy 8-bit baseline, matching the old fashion_small architecture
  fashion_cnn_8_bit_a - single-convolution tiny CNN
  fashion_cnn_8_bit_b - two-convolution tiny CNN
  fashion_skip_8_bit  - tiny CNN with a residual skip connection
"""

from __future__ import annotations

from pathlib import Path

import keras.layers
import numpy as np
from keras.datasets import fashion_mnist
from keras.models import Model, save_model
from keras.optimizers import Adam
from keras.utils import to_categorical
from qkeras import QActivation, QConv2DBatchnorm, QDense, quantized_bits, quantized_relu

NB_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 10
VALIDATION_SPLIT = 0.1

q_bits_input = quantized_bits(bits=8, integer=0, keep_negative=False, symmetric=False, alpha=1)
q_bits_weights = quantized_bits(bits=8, integer=0, keep_negative=True, symmetric=True, alpha=1)
q_bits_bias = quantized_bits(bits=8, integer=0, keep_negative=True, symmetric=True, alpha=1)
q_activation = quantized_relu(
    bits=8,
    integer=0,
    negative_slope=0,
    use_stochastic_rounding=False,
    is_quantized_clip=True,
    use_ste=True,
)


def build_dense_8_bit(input_shape: tuple[int, int, int], name: str) -> Model:
    """Build the dense-heavy 8-bit Fashion-MNIST baseline.

    Args:
        input_shape: Input image shape without the batch dimension.
        name: Keras model name.

    Returns:
        Uncompiled QKeras model.
    """
    x_in = keras.layers.Input(shape=input_shape, name="fashion_input")
    x = QActivation(activation=q_bits_input, name="input_quant")(x_in)

    x = QConv2DBatchnorm(
        filters=8,
        kernel_size=3,
        padding="same",
        kernel_quantizer=q_bits_weights,
        bias_quantizer=q_bits_bias,
        name="conv1_bn",
    )(x)
    x = QActivation(activation=q_activation, name="act1")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    x = keras.layers.Flatten(name="flatten")(x)
    x = QDense(
        units=NB_CLASSES,
        use_bias=False,
        kernel_quantizer=q_bits_weights,
        name="output_dense",
    )(x)
    x = keras.layers.Activation("softmax", name="softmax")(x)
    return Model(inputs=x_in, outputs=x, name=name)


def build_cnn_8_bit_a(input_shape: tuple[int, int, int], name: str) -> Model:
    """Build a single-convolution 8-bit Fashion-MNIST model.

    Args:
        input_shape: Input image shape without the batch dimension.
        name: Keras model name.

    Returns:
        Uncompiled QKeras model.
    """
    x_in = keras.layers.Input(shape=input_shape, name="fashion_input")
    x = QActivation(activation=q_bits_input, name="input_quant")(x_in)

    x = QConv2DBatchnorm(
        filters=5,
        kernel_size=3,
        padding="same",
        kernel_quantizer=q_bits_weights,
        bias_quantizer=q_bits_bias,
        name="conv1_bn",
    )(x)
    x = QActivation(activation=q_activation, name="act1")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    x = keras.layers.Flatten(name="flatten")(x)
    x = QDense(
        units=NB_CLASSES,
        use_bias=False,
        kernel_quantizer=q_bits_weights,
        name="output_dense",
    )(x)
    x = keras.layers.Activation("softmax", name="softmax")(x)
    return Model(inputs=x_in, outputs=x, name=name)


def build_cnn_8_bit_b(input_shape: tuple[int, int, int], name: str) -> Model:
    """Build a two-convolution 8-bit Fashion-MNIST model.

    Args:
        input_shape: Input image shape without the batch dimension.
        name: Keras model name.

    Returns:
        Uncompiled QKeras model.
    """
    x_in = keras.layers.Input(shape=input_shape, name="fashion_input")
    x = QActivation(activation=q_bits_input, name="input_quant")(x_in)

    x = QConv2DBatchnorm(
        filters=4,
        kernel_size=3,
        padding="same",
        kernel_quantizer=q_bits_weights,
        bias_quantizer=q_bits_bias,
        name="conv1_bn",
    )(x)
    x = QActivation(activation=q_activation, name="act1")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    x = QConv2DBatchnorm(
        filters=16,
        kernel_size=3,
        padding="same",
        kernel_quantizer=q_bits_weights,
        bias_quantizer=q_bits_bias,
        name="conv2_bn",
    )(x)
    x = QActivation(activation=q_activation, name="act2")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    x = keras.layers.Flatten(name="flatten")(x)
    x = QDense(
        units=NB_CLASSES,
        use_bias=False,
        kernel_quantizer=q_bits_weights,
        name="output_dense",
    )(x)
    x = keras.layers.Activation("softmax", name="softmax")(x)
    return Model(inputs=x_in, outputs=x, name=name)


def build_skip_8_bit(input_shape: tuple[int, int, int], name: str) -> Model:
    """Build an 8-bit Fashion-MNIST model with a residual skip connection.

    Args:
        input_shape: Input image shape without the batch dimension.
        name: Keras model name.

    Returns:
        Uncompiled QKeras model.
    """
    x_in = keras.layers.Input(shape=input_shape, name="fashion_input")
    x = QActivation(activation=q_bits_input, name="input_quant")(x_in)

    x = QConv2DBatchnorm(
        filters=5,
        kernel_size=3,
        padding="same",
        kernel_quantizer=q_bits_weights,
        bias_quantizer=q_bits_bias,
        name="conv1_bn",
    )(x)
    x = QActivation(activation=q_activation, name="act1")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    x_skip = x

    x = QConv2DBatchnorm(
        filters=5,
        kernel_size=3,
        padding="same",
        kernel_quantizer=q_bits_weights,
        bias_quantizer=q_bits_bias,
        name="conv3_bn",
    )(x)
    x = keras.layers.Add(name="skip_add")([x_skip, x])
    x = QActivation(activation=q_activation, name="act2")(x)

    x = keras.layers.Flatten(name="flatten")(x)
    x = QDense(
        units=NB_CLASSES,
        use_bias=False,
        kernel_quantizer=q_bits_weights,
        name="output_dense",
    )(x)
    x = keras.layers.Activation("softmax", name="softmax")(x)
    return Model(inputs=x_in, outputs=x, name=name)


VARIANTS = {
    "fashion_dense_8_bit": build_dense_8_bit,
    "fashion_cnn_8_bit_a": build_cnn_8_bit_a,
    "fashion_cnn_8_bit_b": build_cnn_8_bit_b,
    "fashion_skip_8_bit": build_skip_8_bit,
}


def main() -> None:
    """Train all Fashion-MNIST examples and save models plus test vectors."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.astype("float32")[..., np.newaxis] / 256.0
    x_test = x_test.astype("float32")[..., np.newaxis] / 256.0
    y_train = to_categorical(y_train, NB_CLASSES)
    y_test = to_categorical(y_test, NB_CLASSES)

    demo_inputs = x_test[:1000]
    input_shape = x_train.shape[1:]

    for name, builder in VARIANTS.items():
        print(f"\n{'=' * 50}")
        print(f"Training {name}")
        print(f"{'=' * 50}")
        model = builder(input_shape, name)
        model.summary()
        model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=0.001),
            metrics=["accuracy"],
        )
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=True,
            validation_split=VALIDATION_SPLIT,
        )
        score = model.evaluate(x_test, y_test, verbose=0)
        print(f"{name} test loss: {score[0]:.4f}, test accuracy: {score[1]:.4f}")

        save_model(model, models_dir / f"{name}.h5")
        np.save(models_dir / f"X_test_{name}.npy", demo_inputs)
        np.save(models_dir / f"labels_test_{name}.npy", y_test[: len(demo_inputs)])
        np.save(models_dir / f"y_test_{name}.npy", model.predict(demo_inputs, verbose=0))


if __name__ == "__main__":
    main()
