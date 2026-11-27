#
# Copyright (c) 2026 Dan McLeran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#


"""Train a small residual CNN on the UCI optical digits in Keras, convert it
to an int8 TensorFlow Lite model, and generate the tinymind model the C++
example runs.

    make regenerate PY=<python with numpy, tensorflow, tflite-micro>

Writes, next to this script:
  models/digits_int8.tflite
                 the int8 model TFLiteConverter writes (full-integer
                 quantization, int8 input and output)
  weights.hpp    its int8 weights, int32 biases and requantizers
                 (apps/import_tflite)
  model.hpp      layer objects, memory plan and forward()
  reference.hpp  the input quantization table; for every test digit, the class
                 the float Keras model predicts and the ten int8 outputs that
                 TensorFlow Lite for Microcontrollers computes for it

The C++ driver requires the generated model to reproduce every one of TFLM's
outputs. The model, the generated headers and the reference are committed,
so building and running the example needs no Python.
"""

import os
import sys

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402
from tflite_micro.python.tflite_micro import runtime as tflm_runtime  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "optical_digits")
sys.path.insert(0, os.path.join(HERE, "..", "..", "apps", "import_tflite"))
import tinymind_import_tflite as tt  # noqa: E402

SEED = 1
EPOCHS = 60
NAMESPACE = "digits"
CALIBRATION_DIGITS = 512


def load(name):
    raw = np.loadtxt(os.path.join(DATA, name), delimiter=",", dtype=np.int64)
    return raw[:, :64], raw[:, 64]


def to_float(pixels):
    return (pixels.astype(np.float32) / 16.0).reshape(-1, 8, 8, 1)


def digit_net():
    """Conv 3x3 -> residual block (depthwise 3x3, pointwise 1x1, added back)
    -> max pool -> conv 3x3 -> global average pool -> dense -> softmax."""
    L = tf.keras.layers
    x = L.Input((8, 8, 1))
    h = L.Conv2D(8, 3, padding="same", activation="relu")(x)
    r = L.DepthwiseConv2D(3, padding="same", activation="relu6")(h)
    r = L.Conv2D(8, 1)(r)
    h = L.Add()([h, r])
    h = L.MaxPooling2D(2)(h)
    h = L.Conv2D(16, 3, padding="same", activation="relu")(h)
    h = L.GlobalAveragePooling2D()(h)
    h = L.Dense(10)(h)
    y = L.Softmax()(h)
    return tf.keras.Model(x, y)


def main():
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    train_px, train_y = load("optdigits.tra")
    test_px, test_y = load("optdigits.tes")
    x_train, x_test = to_float(train_px), to_float(test_px)

    model = digit_net()
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-3), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(x_train, train_y, batch_size=64, epochs=EPOCHS, verbose=0, shuffle=True)
    float_pred = model.predict(x_test, verbose=0).argmax(1)
    print(f"Keras float test accuracy: {np.mean(float_pred == test_y):.4f}")

    calib = x_train[np.random.default_rng(SEED).permutation(len(x_train))[:CALIBRATION_DIGITS]]

    def representative():
        for i in range(len(calib)):
            yield [calib[i:i + 1]]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    blob = conv.convert()
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    tflite_path = os.path.join(HERE, "models", "digits_int8.tflite")
    with open(tflite_path, "wb") as f:
        f.write(blob)

    layers, generated = tt.import_tflite_model_with_code(
        tflite_path, os.path.join(HERE, "weights.hpp"), os.path.join(HERE, "model.hpp"), NAMESPACE)
    print(f"{len(layers)} layers: {[l.op_type for l in layers]}")
    print(f"arena {generated.plan.arena_size} B (unshared {generated.plan.unshared_size} B, "
          f"busiest layer {generated.plan.lower_bound} B), weights {generated.weight_bytes} B")

    # The input grid: pixel p in 0..16 is the float p / 16, quantized as
    # TFLite's QUANTIZE does (round half away from zero).
    first = layers[0]
    table = [max(-128, min(127, tt.tflite_round(float(np.float32(p / 16.0) / np.float32(first.input_scale)))
                           + first.input_zp)) for p in range(17)]

    # TFLM's int8 outputs for every test digit, on the same int8 inputs.
    interp = tflm_runtime.Interpreter.from_bytes(blob)
    tflm = np.zeros((len(test_y), 10), dtype=np.int64)
    for n in range(len(test_y)):
        q = np.array([table[v] for v in test_px[n]], dtype=np.int8).reshape(1, 8, 8, 1)
        interp.set_input(q, 0)
        interp.invoke()
        tflm[n] = np.array(interp.get_output(0)).reshape(-1)
    tflm_pred = tflm.argmax(1)
    print(f"TFLM int8 test accuracy: {np.mean(tflm_pred == test_y):.4f}; "
          f"same class as Keras on {int(np.sum(tflm_pred == float_pred))} of {len(test_y)}")

    def arr(name, values, ctype, per_line=32):
        rows = [", ".join(str(int(v)) for v in values[i:i + per_line]) for i in range(0, len(values), per_line)]
        return f"    constexpr {ctype} {name}[{len(values)}] = {{\n        " + ",\n        ".join(rows) + "\n    };\n"

    with open(os.path.join(HERE, "reference.hpp"), "w") as f:
        f.write(tt.ti.copyright_block("examples/tflite_digits_int8/train.py")
                + "\n#pragma once\n\n#include <cstddef>\n#include <cstdint>\n\n"
                f"namespace {NAMESPACE} {{\n"
                "    // Pixel value 0..16 -> the model's int8 input grid.\n"
                + arr("kPixelToInput", table, "int8_t", 17)
                + f"    constexpr std::size_t kTestCount = {len(test_y)};\n"
                "    // The class the float Keras model predicts for each test digit.\n"
                + arr("kKerasPrediction", float_pred.tolist(), "uint8_t")
                + "    // TFLM's ten int8 outputs for each test digit, digit after digit.\n"
                + arr("kTflmOutput", tflm.reshape(-1).tolist(), "int8_t", 20)
                + f"}} // namespace {NAMESPACE}\n")


if __name__ == "__main__":
    main()
