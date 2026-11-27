"""
Copyright (c) 2026 Dan McLeran

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Tests for the TFLite importer. Needs a Python with numpy, tensorflow and
tflite-micro (TFLM's Python interpreter), and a host g++; use a venv, never
the system Python:

    make test PY=<python with numpy, tensorflow, tflite-micro>

Every case is judged against TensorFlow Lite for Microcontrollers itself:

  * the flatbuffer reader returns what TensorFlow's generated schema reader
    returns, field by field, for every tensor and operator of every model;
  * the LOGISTIC and TANH tables and the MEAN requantizer equal TFLM's output
    for every int8 input, over many calibrations;
  * whole models (Keras, converted with full-integer int8 quantization) are
    imported, generated as model.hpp, compiled freestanding (FLOAT=0, STD=0)
    and run on random int8 inputs; every output must equal TFLM's, bit for
    bit.

The models are chosen so a wrong mapping cannot pass: SAME padding with an
even kernel and a stride (catches the pad split), a non-square kernel with
unequal strides (catches KH/KW and stride order), a depthwise convolution
(catches the [1][KH][KW][C] -> [C][KH][KW] transpose), fused RELU and RELU6
(catch a missing clamp), a residual ADD, a Flatten into a dense layer (catches
a wrong flattening order), and a softmax and a sigmoid at the output.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import tflite_flatbuffer as fb  # noqa: E402
import tinymind_import_tflite as tt  # noqa: E402

try:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    from tensorflow.lite.python import schema_py_generated as schema
    from tflite_micro.python.tflite_micro import runtime as tflm_runtime
    HAVE_STACK = True
except ImportError:  # pragma: no cover - exercised only without the deps
    HAVE_STACK = False

GXX = shutil.which("g++")


# ----------------------------------------------------------------------------
# Model building and the TFLM reference.
# ----------------------------------------------------------------------------


def randomize(model, seed):
    """Random weights and nonzero biases: an untrained Keras layer's zero
    bias is dropped by the converter, which would leave the bias path
    untested."""
    rng = np.random.default_rng(1000 + seed)
    for layer in model.layers:
        ws = layer.get_weights()
        if ws:
            layer.set_weights([(rng.standard_normal(w.shape) * (0.3 if w.ndim == 1 else 0.5)).astype(w.dtype)
                               for w in ws])
    return model


def convert(model, seed=0, input_spread=1.0, int_io=True, calib=96):
    """Full-integer int8 conversion with a random representative set."""
    randomize(model, seed)
    rng = np.random.default_rng(seed)
    shape = tuple(model.input_shape[1:])

    def representative():
        for _ in range(calib):
            yield [(input_spread * rng.standard_normal((1,) + shape)).astype(np.float32)]

    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = representative
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if int_io:
        c.inference_input_type = tf.int8
        c.inference_output_type = tf.int8
    return c.convert()


def tflm_run(blob, x):
    it = tflm_runtime.Interpreter.from_bytes(blob)
    it.set_input(x, 0)
    it.invoke()
    return np.array(it.get_output(0))


def tflm_int8_outputs(blob, q_inputs, float_io):
    """TFLM's int8 outputs for int8 inputs; for a float-I/O model the input is
    dequantized exactly (so QUANTIZE returns it) and the output requantized
    exactly (so DEQUANTIZE is undone)."""
    m = fb.read_tflite(blob)
    outs = []
    if float_io:
        qi = m.tensors[m.operators[0].outputs[0]]
        qo = m.tensors[m.operators[-1].inputs[0]]
    for q in q_inputs:
        if float_io:
            x = ((q.astype(np.float32) - qi.zero_points[0]) * np.float32(qi.scales[0])).astype(np.float32)
            y = tflm_run(blob, x.reshape(qi.shape))
            outs.append(np.rint(y / np.float32(qo.scales[0])).astype(np.int64).reshape(-1) + qo.zero_points[0])
        else:
            shape = m.tensors[m.inputs[0]].shape
            outs.append(tflm_run(blob, q.reshape(shape).astype(np.int8)).astype(np.int64).reshape(-1))
    return np.array(outs)


MODEL_DRIVER = r"""
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "@HEADER@"

int main()
{
    typedef @NS@::ModelActivationType T;
    static T input[@NS@::kModelInputSize];
    static T output[@NS@::kModelOutputSize];
    alignas(@NS@::kModelArenaAlignment) static T arena[@NS@::kModelArenaSize > 0 ? @NS@::kModelArenaSize : 1];
    for (;;)
    {
        // Garbage in the arena: the model must not read a byte it did not write.
        std::memset(arena, 0x5A, sizeof(arena));
        for (std::size_t i = 0; i < @NS@::kModelInputSize; ++i)
        {
            int v;
            if (std::scanf("%d", &v) != 1) { return 0; }
            input[i] = static_cast<T>(v);
        }
        @NS@::forward(input, output, arena);
        for (std::size_t i = 0; i < @NS@::kModelOutputSize; ++i)
        {
            std::printf("%d ", static_cast<int>(output[i]));
        }
        std::printf("\n");
    }
}
"""


def build_model(blob, workdir, tag, edit=None, **kwargs):
    path = os.path.join(workdir, f"{tag}.tflite")
    with open(path, "wb") as f:
        f.write(blob)
    weights = os.path.join(workdir, f"{tag}_weights.hpp")
    header = os.path.join(workdir, f"{tag}_model.hpp")
    if edit is None:
        layers, gen = tt.import_tflite_model_with_code(path, weights, header, tag, **kwargs)
    else:
        layers = tt.import_tflite_model(path, weights, tag)
        edit(layers)
        gen = tt.cg.emit_model_header(header, tag, layers, os.path.basename(weights), generator=tt.GENERATOR,
                                      **kwargs)
    src = os.path.join(workdir, f"{tag}_driver.cpp")
    with open(src, "w") as f:
        f.write(MODEL_DRIVER.replace("@HEADER@", os.path.basename(header)).replace("@NS@", tag))
    exe = os.path.join(workdir, f"{tag}_driver")
    subprocess.run([GXX, "-std=c++14", "-O1", "-Wall", "-Wextra", "-Werror", "-Wpedantic",
                    "-DTINYMIND_ENABLE_QUANTIZATION=1", "-DTINYMIND_ENABLE_FLOAT=0", "-DTINYMIND_ENABLE_STD=0",
                    "-I" + os.path.join(REPO, "cpp"), "-I" + os.path.join(REPO, "cpp", "include"),
                    "-I" + os.path.join(REPO, "include"), "-I" + workdir, src, "-o", exe], check=True)
    return exe, layers, gen


def run_exe(exe, q_inputs):
    stdin = "".join(" ".join(str(int(v)) for v in q.reshape(-1)) + "\n" for q in q_inputs)
    out = subprocess.run([exe], input=stdin, capture_output=True, text=True, check=True).stdout
    return np.array([[int(v) for v in row.split()] for row in out.strip().split("\n")])


def random_inputs(blob, n, seed):
    """Uniform int8 inputs and inputs drawn from the calibration's spread."""
    m = fb.read_tflite(blob)
    t = m.tensors[m.inputs[0]]
    if t.type == fb.FLOAT32:
        t = m.tensors[m.operators[0].outputs[0]]
    rng = np.random.default_rng(seed)
    size = int(np.prod(t.shape))
    qs = []
    for i in range(n):
        if i % 2:
            q = np.rint(rng.standard_normal(size) / t.scales[0]) + t.zero_points[0]
        else:
            q = rng.integers(-128, 128, size=size)
        qs.append(np.clip(q, -128, 127).astype(np.int64))
    return qs


# ----------------------------------------------------------------------------
# Keras models.
# ----------------------------------------------------------------------------


def residual_cnn():
    L = tf.keras.layers
    # 11 rows at stride 2 and 10 columns at stride 1 with a 4x4 kernel: both
    # SAME splits are uneven (1 before, 2 after), so a swapped split shows.
    x = L.Input((11, 10, 3))
    y = L.Conv2D(8, (4, 4), strides=(2, 1), padding="same", activation="relu")(x)
    z = L.DepthwiseConv2D(3, padding="same", activation="relu6")(y)
    z = L.Conv2D(8, 1)(z)
    y = L.Add()([y, z])
    y = L.MaxPooling2D(2)(y)
    y = L.AveragePooling2D((1, 2), strides=(1, 1))(y)
    y = L.Flatten()(y)
    y = L.Dense(16, activation="relu")(y)
    y = L.Dense(5)(y)
    y = L.Softmax()(y)
    return tf.keras.Model(x, y)


def gap_sigmoid():
    L = tf.keras.layers
    x = L.Input((9, 7, 2))
    y = L.Conv2D(6, (3, 5), strides=(2, 1), activation="relu")(x)
    y = L.DepthwiseConv2D((2, 3), strides=(1, 2))(y)
    y = L.GlobalAveragePooling2D()(y)
    y = L.Dense(3, activation="sigmoid")(y)
    return tf.keras.Model(x, y)


def mlp_tanh():
    L = tf.keras.layers
    x = L.Input((16,))
    y = L.Dense(12, activation="tanh")(x)
    y = L.Dense(8, activation="relu")(y)
    y = L.Dense(4)(y)
    return tf.keras.Model(x, y)


def global_avgpool_relu():
    L = tf.keras.layers
    x = L.Input((8, 8, 4))
    y = L.Conv2D(4, 3, padding="same")(x)
    y = L.ReLU()(y)
    y = L.AveragePooling2D(8)(y)
    y = L.Reshape((4,))(y)
    y = L.Dense(2)(y)
    return tf.keras.Model(x, y)


# ----------------------------------------------------------------------------
# Tests.
# ----------------------------------------------------------------------------


class ArithmeticMirrors(unittest.TestCase):
    """Runs without TensorFlow: the integer helpers on hand-checked values."""

    def test_quantize_multiplier(self):
        self.assertEqual(tt.quantize_multiplier_tflite(0.5), (1 << 30, 0))
        self.assertEqual(tt.quantize_multiplier_tflite(1.0), (1 << 30, 1))
        self.assertEqual(tt.quantize_multiplier_tflite(0.0), (0, 0))
        q, s = tt.quantize_multiplier_tflite(0.0123)
        self.assertAlmostEqual(q * 2.0 ** (s - 31), 0.0123, places=9)

    def test_multiply_by_quantized_multiplier(self):
        # 0.5 is one doubling high multiply, which rounds half up; 0.25 adds
        # a rounding right shift (half away from zero), so TFLite rounds
        # twice: 5 * 0.25 -> round(2.5) = 3 -> round(1.5) = 2.
        q, s = tt.quantize_multiplier_tflite(0.5)
        for x in range(-1000, 1001):
            self.assertEqual(tt.multiply_by_quantized_multiplier(x, q, s), int(np.floor(x / 2 + 0.5)), x)
        q, s = tt.quantize_multiplier_tflite(0.25)
        for x, want in ((5, 2), (6, 2), (7, 2), (-5, -1), (-6, -2), (-7, -2), (1000, 250)):
            self.assertEqual(tt.multiply_by_quantized_multiplier(x, q, s), want, x)

    def test_activation_range(self):
        self.assertEqual(tt.activation_range(fb.ACT_NONE, 0.1, 3, "t"), (-128, 127))
        self.assertEqual(tt.activation_range(fb.ACT_RELU, 0.1, 3, "t"), (3, 127))
        self.assertEqual(tt.activation_range(fb.ACT_RELU6, 0.1, -10, "t"), (-10, 50))
        self.assertEqual(tt.activation_range(fb.ACT_RELU_N1_TO_1, 0.01, 0, "t"), (-100, 100))
        with self.assertRaises(ValueError):
            tt.activation_range(fb.ACT_TANH, 0.1, 0, "t")

    def test_fixed_point_functions(self):
        # Spot values against the real functions (raw Q4.27 in, Q0.31 out).
        for v in (-7.5, -2.0, -0.3, 0.0, 0.3, 1.0, 4.0, 7.9):
            raw = int(round(v * (1 << 27)))
            self.assertAlmostEqual(tt.fixed_point_logistic(raw, 4) / 2.0 ** 31, 1 / (1 + np.exp(-v)), places=6)
            self.assertAlmostEqual(tt.fixed_point_tanh(raw, 4) / 2.0 ** 31, np.tanh(v), places=6)


@unittest.skipUnless(HAVE_STACK, "needs tensorflow and tflite-micro")
class FlatbufferReader(unittest.TestCase):
    """tflite_flatbuffer returns what TensorFlow's schema reader returns."""

    def check(self, blob):
        mine = fb.read_tflite(blob)
        ref = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(blob, 0))
        self.assertEqual(mine.version, ref.version)
        sg = ref.subgraphs[0]
        self.assertEqual(mine.inputs, list(sg.inputs))
        self.assertEqual(mine.outputs, list(sg.outputs))
        self.assertEqual(len(mine.tensors), len(sg.tensors))
        for t, r in zip(mine.tensors, sg.tensors):
            self.assertEqual(t.name, r.name.decode())
            self.assertEqual(t.shape, [] if r.shape is None else [int(v) for v in r.shape])
            self.assertEqual(t.type, r.type)
            data = ref.buffers[r.buffer].data
            if data is None or len(data) == 0:
                self.assertIsNone(t.data)
            else:
                self.assertEqual(t.data.tobytes(), bytes(np.asarray(data, dtype=np.uint8)))
            q = r.quantization
            scales = [] if q is None or q.scale is None else [float(v) for v in q.scale]
            zps = [] if q is None or q.zeroPoint is None else [int(v) for v in q.zeroPoint]
            self.assertEqual(t.scales, scales)
            self.assertEqual(t.zero_points, zps)
            self.assertEqual(t.quantized_dimension, 0 if q is None else q.quantizedDimension)
        names = {v: k for k, v in schema.BuiltinOperator.__dict__.items() if not k.startswith("_")}
        self.assertEqual(len(mine.operators), len(sg.operators))
        for o, r in zip(mine.operators, sg.operators):
            code = ref.operatorCodes[r.opcodeIndex]
            self.assertEqual(o.op, names[max(code.builtinCode, code.deprecatedBuiltinCode)])
            self.assertEqual(o.inputs, [int(v) for v in r.inputs])
            self.assertEqual(o.outputs, [int(v) for v in r.outputs])
            opts = r.builtinOptions
            for key, value in o.options.items():
                camel = key.split("_")[0] + "".join(w.title() for w in key.split("_")[1:])
                camel = camel.replace("Factor", "Factor")
                ref_value = getattr(opts, camel)
                if isinstance(value, float):
                    self.assertEqual(np.float32(value), np.float32(ref_value), (o.op, key))
                else:
                    self.assertEqual(value, ref_value, (o.op, key))

    def test_reader_matches_tensorflow(self):
        for build, seed in ((residual_cnn, 1), (gap_sigmoid, 2), (mlp_tanh, 3), (global_avgpool_relu, 4)):
            blob = convert(build(), seed=seed)
            self.check(blob)
        self.check(convert(mlp_tanh(), seed=5, int_io=False))

    def test_rejects_other_files(self):
        with self.assertRaises(ValueError):
            fb.read_tflite(b"\x00" * 64)


@unittest.skipUnless(HAVE_STACK, "needs tensorflow and tflite-micro")
class ExhaustiveOps(unittest.TestCase):
    """The tables and parameters the importer computes equal TFLM's output for
    every int8 input."""

    def single_op(self, layer_fn, shape, spread, seed):
        x = tf.keras.layers.Input(shape)
        return convert(tf.keras.Model(x, layer_fn(x)), seed=seed, input_spread=spread)

    def test_logistic_and_tanh_tables(self):
        q = np.arange(-128, 128, dtype=np.int8).reshape(1, 256)
        cases = 0
        for kind in ("sigmoid", "tanh"):
            for k, spread in enumerate((0.05, 0.3, 1.0, 2.5, 6.0, 15.0, 60.0)):
                blob = self.single_op(lambda x: tf.keras.layers.Activation(kind)(x), (256,), spread, 10 + k)
                layers = tt.parse_tflite_model(blob)
                self.assertEqual(len(layers), 1)
                want = tflm_run(blob, q).reshape(-1).astype(int).tolist()
                self.assertEqual(layers[0].lut, want, (kind, spread, layers[0].input_scale, layers[0].input_zp))
                cases += 1
        self.assertEqual(cases, 14)

    def test_mean(self):
        for k, shape in enumerate(((4, 4, 3), (3, 5, 2), (6, 6, 4), (1, 7, 5), (16, 16, 2), (5, 5, 3))):
            for spread in (0.2, 3.0):
                blob = self.single_op(lambda x: tf.keras.layers.GlobalAveragePooling2D()(x), shape, spread, 30 + k)
                layers = tt.parse_tflite_model(blob)
                self.assertEqual([l.op_type for l in layers], ["Mean"])
                rng = np.random.default_rng(k)
                lay = layers[0]
                for _ in range(40):
                    q = rng.integers(-128, 128, size=(1,) + shape).astype(np.int8)
                    want = tflm_run(blob, q).reshape(-1).astype(int)
                    acc = (q.reshape(-1, shape[2]).astype(np.int64) - lay.input_zp).sum(axis=0)
                    got = [max(-128, min(127, tt.multiply_by_quantized_multiplier(
                        int(a), lay.multipliers[0], -lay.shifts[0]) + lay.output_zp)) for a in acc]
                    self.assertEqual(got, want.tolist(), (shape, spread))


@unittest.skipUnless(HAVE_STACK, "needs tensorflow and tflite-micro")
@unittest.skipIf(GXX is None, "needs g++")
class EndToEnd(unittest.TestCase):
    """The generated model.hpp computes TFLM's outputs bit for bit."""

    def check(self, model, seed, ops, float_io=False, n=48, **kwargs):
        blob = convert(model, seed=seed, int_io=not float_io)
        names = [o.op for o in fb.read_tflite(blob).operators]
        for op in ops:
            self.assertIn(op, names)
        qs = random_inputs(blob, n, seed)
        want = tflm_int8_outputs(blob, qs, float_io)
        with tempfile.TemporaryDirectory() as d:
            exe, layers, gen = build_model(blob, d, f"m{seed}", **kwargs)
            got = run_exe(exe, qs)
        self.assertEqual(got.shape, want.shape)
        mismatches = int((got != want).sum())
        self.assertEqual(mismatches, 0, f"{mismatches} of {want.size} outputs differ from TFLM")
        self.assertGreater(len(np.unique(want)), 2, "outputs are degenerate; the test proves little")
        self.assertTrue(all(l.bias is not None for l in layers if l.weight is not None),
                        "a weight layer has no bias; the bias path is untested")
        return layers, gen

    def test_residual_cnn(self):
        layers, gen = self.check(residual_cnn(), 1, ["CONV_2D", "DEPTHWISE_CONV_2D", "ADD", "MAX_POOL_2D",
                                                       "AVERAGE_POOL_2D", "RESHAPE", "FULLY_CONNECTED",
                                                       "SOFTMAX"])
        self.assertEqual((layers[0].act_min, layers[0].act_max), (layers[0].output_zp, 127))
        self.assertLess(gen.plan.arena_size, gen.plan.unshared_size)

    def test_gap_sigmoid(self):
        self.check(gap_sigmoid(), 2, ["CONV_2D", "DEPTHWISE_CONV_2D", "MEAN", "FULLY_CONNECTED", "LOGISTIC"])

    def test_mlp_tanh_float_io(self):
        layers, _ = self.check(mlp_tanh(), 3, ["QUANTIZE", "FULLY_CONNECTED", "TANH", "DEQUANTIZE"],
                               float_io=True)
        self.assertEqual([l.op_type for l in layers], ["Dense", "Tanh", "Dense", "Dense"])

    def test_global_avgpool_relu(self):
        self.check(global_avgpool_relu(), 4, ["CONV_2D", "AVERAGE_POOL_2D", "FULLY_CONNECTED"])

    def test_clamp_is_emitted(self):
        # Converted models' fused-activation clamps coincide with the int8
        # range (the output grid starts at the clamp), so narrow one by hand:
        # the generated output must be TFLM's, clipped.
        blob = convert(mlp_tanh(), seed=6)
        qs = random_inputs(blob, 24, 6)
        want = tflm_int8_outputs(blob, qs, False)
        lo, hi = int(np.percentile(want, 25)), int(np.percentile(want, 75))

        def narrow(layers):
            layers[-1].act_min, layers[-1].act_max = lo, hi

        with tempfile.TemporaryDirectory() as d:
            exe, _, _ = build_model(blob, d, "clamp", edit=narrow)
            got = run_exe(exe, qs)
            with open(os.path.join(d, "clamp_model.hpp")) as f:
                self.assertIn(f", {lo}, {hi}}}", f.read())
        self.assertTrue(lo < hi)
        np.testing.assert_array_equal(got, np.clip(want, lo, hi))

    def test_in_place_off(self):
        self.check(residual_cnn(), 5, ["ADD"], in_place=False, n=12)


# The MLPerf Tiny reference models (github.com/mlcommons/tiny, Apache 2.0,
# MLCommons), as benchmarks/framework_comparison_f411's `make mlperf-models`
# downloads and pins them. Not committed here; the test runs when they are
# present (in that cache, or in $TINYMIND_MLPERF_TFLITE_DIR).
MLPERF_MODELS = {
    "kws_ref_model.tflite": "aeea436800704fce17b17292e4412630ad856e9d777c044c64ef748a880bd0ae",
    "pretrainedResnet_quant.tflite": "3c002613d1b2475eb51dd78dfb85a546c8ae658dee71cf6ade43b022fe205415",
}
MLPERF_DIR = os.environ.get("TINYMIND_MLPERF_TFLITE_DIR", os.path.join(
    REPO, "benchmarks", "framework_comparison_f411", "gen", "mlperf", "cache"))


@unittest.skipUnless(HAVE_STACK, "needs tensorflow and tflite-micro")
@unittest.skipIf(GXX is None, "needs g++")
@unittest.skipUnless(all(os.path.exists(os.path.join(MLPERF_DIR, f)) for f in MLPERF_MODELS),
                     "MLPerf Tiny models not downloaded (make -C benchmarks/framework_comparison_f411 mlperf-models)")
class MLPerfTinyModels(unittest.TestCase):
    """The MLPerf Tiny keyword-spotting DS-CNN and ResNet-8, imported with no
    hand-written code, compute TFLM's outputs bit for bit."""

    def test_reference_models(self):
        import hashlib
        for name, sha in MLPERF_MODELS.items():
            with open(os.path.join(MLPERF_DIR, name), "rb") as f:
                blob = f.read()
            self.assertEqual(hashlib.sha256(blob).hexdigest(), sha, name)
            qs = random_inputs(blob, 16, 1)
            want = tflm_int8_outputs(blob, qs, False)
            with tempfile.TemporaryDirectory() as d:
                exe, layers, gen = build_model(blob, d, "mlperf")
                got = run_exe(exe, qs)
            self.assertEqual(int((got != want).sum()), 0, name)
            self.assertGreater(len(np.unique(want)), 10, name)


@unittest.skipUnless(HAVE_STACK, "needs tensorflow and tflite-micro")
class Refusals(unittest.TestCase):
    """What the importer cannot run exactly is refused, never skipped."""

    def refuses(self, model, pattern, **kwargs):
        blob = convert(model, **kwargs)
        with self.assertRaisesRegex(ValueError, pattern):
            tt.parse_tflite_model(blob)

    def test_dilated_conv(self):
        L = tf.keras.layers
        x = L.Input((10, 10, 2))
        # The converter lowers a dilated convolution to SPACE_TO_BATCH_ND,
        # CONV_2D, BATCH_TO_SPACE_ND.
        self.refuses(tf.keras.Model(x, L.Conv2D(3, 3, dilation_rate=2)(x)), "unsupported operator SPACE_TO_BATCH_ND")

    def test_depth_multiplier(self):
        L = tf.keras.layers
        x = L.Input((6, 6, 2))
        self.refuses(tf.keras.Model(x, L.DepthwiseConv2D(3, depth_multiplier=2)(x)), "depth multiplier")

    def test_padded_pooling(self):
        L = tf.keras.layers
        x = L.Input((5, 5, 2))
        self.refuses(tf.keras.Model(x, L.MaxPooling2D(2, padding="same")(x)), "padded pooling")

    def test_unsupported_op(self):
        L = tf.keras.layers
        x = L.Input((8,))
        y = L.Dense(8)(x)
        self.refuses(tf.keras.Model(x, L.Multiply()([y, y])), "unsupported operator MUL")


if __name__ == "__main__":
    unittest.main()
