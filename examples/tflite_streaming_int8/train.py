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

"""Train a small streaming int8 CNN in Keras that detects a bearing fault in a
machine's vibration, convert it to an int8 TensorFlow Lite model, and
generate the two tinymind models the C++ example runs: forward() over a
whole window of frames and step(), which takes one new frame per call.

    make regenerate PY=<python with numpy, tensorflow, tflite-micro>

The signal is synthetic (see machine()): 3,200 samples per second, a motor
whose speed wanders between 28 and 32 Hz with four harmonics and broadband
noise, and, when the bearing is damaged, impacts at about 100 per second that
ring a damped structural resonance near 900 Hz. Every 20 ms (64 samples) a
128-sample Hann window gives 16 log band energies: one frame. The model reads
32 frames (0.64 s) and gives the probability of a fault.

Writes, next to this script:
  models/streaming_int8.tflite
                    the int8 model TFLiteConverter writes
  weights.hpp       its weights, biases and requantizers (apps/import_tflite)
  model.hpp         forward() over the whole 32-frame window
  model_stream.hpp  step() over one frame, in namespace vibration::stream
  reference.hpp     a 16-second test recording as int8 frames (healthy for 6 s,
                    then a fault that grows to full severity at 14 s), the
                    fault severity at each frame, and TFLM's two int8 outputs
                    for every 32-frame window of it

The C++ driver requires forward() and step() to reproduce every one of TFLM's
outputs. Everything it reads is committed, so building and running the
example needs no Python.
"""

import os
import sys

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402
from tflite_micro.python.tflite_micro import runtime as tflm_runtime  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "apps", "import_tflite"))
import tinymind_import_tflite as tt  # noqa: E402

SEED = 1
FS = 3200          # samples per second
HOP = 64           # samples per frame (20 ms)
WIN = 128          # samples per analysis window
BANDS = 16         # log band energies per frame
FRAMES = 32        # frames per model window (0.64 s)
EPOCHS = 30
NAMESPACE = "vibration"
TEST_SECONDS = 16.0


def machine(seconds, rng, severity):
    """A vibration recording. severity(t) in 0..1 scales the fault impacts at
    time t (seconds)."""
    n = int(seconds * FS)
    t = np.arange(n) / FS
    speed = rng.uniform(28.0, 32.0) + 0.6 * np.sin(2 * np.pi * t / rng.uniform(3.0, 7.0))
    phase = 2 * np.pi * np.cumsum(speed) / FS
    x = np.zeros(n)
    for k, a in ((1, 1.0), (2, 0.45), (3, 0.3), (4, 0.15)):
        x += a * rng.uniform(0.7, 1.3) * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    x += rng.uniform(0.15, 0.3) * rng.standard_normal(n)
    # Bearing fault: an impact every 1/rate seconds (with jitter), each ringing
    # a resonance that decays in a couple of milliseconds.
    rate, resonance = rng.uniform(90.0, 120.0), rng.uniform(800.0, 1000.0)
    impacts = np.zeros(n)
    when = np.arange(0.0, seconds, 1.0 / rate) + rng.uniform(0, 1.0 / rate)
    when += rng.normal(0, 0.05 / rate, len(when))
    idx = (when[(when >= 0) & (when < seconds)] * FS).astype(int)
    impacts[idx] = severity(idx / FS) * rng.uniform(0.6, 1.4, len(idx))
    k = np.arange(64) / FS
    ring = np.exp(-k / 0.002) * np.sin(2 * np.pi * resonance * k)
    x += 1.5 * np.convolve(impacts, ring)[:n]
    return x * 10 ** (rng.uniform(-6.0, 6.0) / 20)


def features(x):
    """One frame of 16 log band energies per HOP samples."""
    window = np.hanning(WIN)
    count = (len(x) - WIN) // HOP + 1
    out = np.zeros((count, BANDS))
    for i in range(count):
        power = np.abs(np.fft.rfft(x[i * HOP:i * HOP + WIN] * window)) ** 2
        out[i] = np.log10(power[1:1 + 4 * BANDS].reshape(BANDS, 4).sum(1) + 1e-6)
    return out


def windows(frames, stride):
    return np.array([frames[i:i + FRAMES] for i in range(0, len(frames) - FRAMES + 1, stride)])


def detector():
    """Separable 1D convolutions along time, valid padding and stride 1, so
    the newest output depends only on the newest frames of each layer: the
    receptive field is exactly the 32-frame window."""
    L = tf.keras.layers
    x = L.Input((FRAMES, 1, BANDS))
    h = L.Conv2D(16, (3, 1), activation="relu")(x)             # 30 frames
    h = L.DepthwiseConv2D((5, 1), activation="relu")(h)        # 26
    h = L.Conv2D(16, 1, activation="relu")(h)
    h = L.DepthwiseConv2D((9, 1), activation="relu")(h)        # 18
    h = L.Conv2D(16, 1, activation="relu")(h)
    h = L.DepthwiseConv2D((18, 1))(h)                          # 1
    h = L.Conv2D(8, 1, activation="relu")(h)
    h = L.Flatten()(h)
    y = L.Softmax()(L.Dense(2)(h))
    return tf.keras.Model(x, y)


def main():
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    rng = np.random.default_rng(SEED)

    # Training: 3-second recordings, half of them with a fault of constant
    # severity, every 32-frame window of each.
    xs, ys = [], []
    for r in range(600):
        faulty = r % 2 == 1
        level = rng.uniform(0.15, 1.0) if faulty else 0.0
        f = features(machine(3.0, rng, lambda t, s=level: np.full(np.shape(t), s)))
        w = windows(f, 2)
        xs.append(w)
        ys.append(np.full(len(w), int(faulty)))
    x_train, y_train = np.concatenate(xs), np.concatenate(ys)
    mean, std = float(x_train.mean()), float(x_train.std())
    x_train = ((x_train - mean) / std).astype(np.float32)[:, :, None, :]

    model = detector()
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-3), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=128, epochs=EPOCHS, verbose=0, shuffle=True)

    calib = x_train[rng.permutation(len(x_train))[:512]]

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
    tflite_path = os.path.join(HERE, "models", "streaming_int8.tflite")
    with open(tflite_path, "wb") as f:
        f.write(blob)

    weights = os.path.join(HERE, "weights.hpp")
    layers, whole = tt.import_tflite_model_with_code(tflite_path, weights, os.path.join(HERE, "model.hpp"),
                                                     NAMESPACE)
    stream = tt.cg.emit_model_header(os.path.join(HERE, "model_stream.hpp"), NAMESPACE, layers, "weights.hpp",
                                     generator=tt.GENERATOR, nested_namespace="stream", streaming=True)
    sp = stream.stream
    print(f"{len(layers)} layers: {[l.op_type for l in layers]}")
    print(f"whole window: input {whole.input_size} B, arena {whole.plan.arena_size} B; streaming: frame "
          f"{stream.input_size} B, arena {stream.plan.arena_size} B, state {sp.state_size} B; "
          f"receptive field {sp.receptive_frames} frames; weights {whole.weight_bytes} B")

    # The test recording, quantized onto the model's input grid as TFLite's
    # QUANTIZE does (round half away from zero).
    def ramp(t):
        return np.clip((np.asarray(t) - 6.0) / 8.0, 0.0, 1.0)

    test_rng = np.random.default_rng(SEED + 100)
    test = (features(machine(TEST_SECONDS, test_rng, ramp)) - mean) / std
    first = layers[0]
    q = np.array([[max(-128, min(127, tt.tflite_round(float(np.float32(v) / np.float32(first.input_scale)))
                                 + first.input_zp)) for v in row] for row in test], dtype=np.int64)
    severity = ramp((np.arange(len(q)) * HOP + WIN) / FS)

    interp = tflm_runtime.Interpreter.from_bytes(blob)
    tflm = []
    for end in range(FRAMES - 1, len(q)):
        interp.set_input(q[end - FRAMES + 1:end + 1].reshape(1, FRAMES, 1, BANDS).astype(np.int8), 0)
        interp.invoke()
        tflm.append(np.array(interp.get_output(0)).reshape(-1))
    tflm = np.array(tflm, dtype=np.int64)
    p_fault = (tflm[:, 1] + 128) / 256.0
    truth = severity[FRAMES - 1:] > 0
    print(f"TFLM on the test recording: fault probability > 0.5 on {np.mean(p_fault[~truth] > 0.5):.3f} of "
          f"healthy windows and {np.mean(p_fault[truth] > 0.5):.3f} of faulty ones")

    def arr(name, values, ctype, per_line=32):
        rows = [", ".join(str(int(v)) for v in values[i:i + per_line]) for i in range(0, len(values), per_line)]
        return f"    constexpr {ctype} {name}[{len(values)}] = {{\n        " + ",\n        ".join(rows) + "\n    };\n"

    with open(os.path.join(HERE, "reference.hpp"), "w") as f:
        f.write(tt.ti.copyright_block("examples/tflite_streaming_int8/train.py")
                + "\n#pragma once\n\n#include <cstddef>\n#include <cstdint>\n\n"
                f"namespace {NAMESPACE} {{\n"
                f"    constexpr std::size_t kSampleRate = {FS};\n"
                f"    constexpr std::size_t kHopSamples = {HOP};\n"
                f"    constexpr std::size_t kWindowSamples = {WIN};\n"
                f"    constexpr std::size_t kFrameCount = {len(q)};\n"
                "    // The test recording: one frame of 16 int8 log band energies, frame after frame.\n"
                + arr("kFrames", q.reshape(-1).tolist(), "int8_t", BANDS)
                + "    // The fault severity (percent) at the end of each frame.\n"
                + arr("kSeverityPercent", np.rint(100 * severity).astype(int).tolist(), "uint8_t")
                + "    // TFLM's two int8 outputs (healthy, fault) for the window ending at\n"
                f"    // frame {FRAMES - 1}, {FRAMES}, ... kFrameCount - 1.\n"
                + arr("kTflmOutput", tflm.reshape(-1).tolist(), "int8_t")
                + f"}} // namespace {NAMESPACE}\n")


if __name__ == "__main__":
    main()
