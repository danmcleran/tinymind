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

Tests for the PyTorch importer (numpy only, no torch).

    python3 -m unittest test_import_pytorch    (from apps/import_pytorch)
"""

import math
import os
import subprocess
import tempfile
import unittest

import numpy as np

from tinymind_import import (LSTM, LSTM_LUT_INPUT_SCALE, BatchNorm2D, Conv2D,
                             Dense, MinMaxObserver, MoE, ReLU, Softmax,
                             calibrate, fuse_layers, generate_model,
                             import_pytorch_model, lstm_float_forward, lround,
                             quantize_weights)

try:
    import torch
except ImportError:  # torch is optional; the parity test below skips.
    torch = None


def conv2d_valid(x, w, b):
    """NHWC input [H, W, C], OHWI weights [F, KH, KW, C], VALID, stride 1."""
    f, kh, kw, _ = w.shape
    oh = x.shape[0] - kh + 1
    ow = x.shape[1] - kw + 1
    y = np.empty((oh, ow, f), dtype=np.float64)
    for i in range(oh):
        for j in range(ow):
            patch = x[i:i + kh, j:j + kw, :]
            y[i, j, :] = np.tensordot(w, patch, axes=([1, 2, 3], [0, 1, 2])) + b
    return y


class ConvBatchNormFusion(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.w = rng.uniform(-0.5, 0.5, size=(4, 3, 3, 2))
        self.b = rng.uniform(-0.1, 0.1, size=4)
        w, b = self.w, self.b
        self.conv = Conv2D(name="conv", weight=w, bias=b, input_name="x",
                           forward=lambda x: conv2d_valid(x, w, b))
        # Scale and shift each channel well away from the conv's own range,
        # so a range measured before the BN cannot pass for one after it.
        self.bn = BatchNorm2D(name="bn", input_name="conv",
                              gamma=np.array([4.0, 0.25, -2.0, 1.0]),
                              beta=np.array([3.0, -1.0, 0.5, 6.0]),
                              mean=np.array([0.1, -0.2, 0.0, 0.3]),
                              variance=np.array([0.5, 2.0, 1.0, 0.25]))
        self.data = [rng.uniform(-1.0, 1.0, size=(6, 6, 2)) for _ in range(8)]

    def bn_reference(self, x):
        bn = self.bn
        sigma = bn.gamma / np.sqrt(bn.variance + bn.eps)
        return (self.conv.forward(x) - bn.mean) * sigma + bn.beta

    def test_fused_weights_compute_conv_then_bn(self):
        (fused,) = fuse_layers([self.conv, self.bn])
        for x in self.data:
            np.testing.assert_allclose(conv2d_valid(x, fused.weight, fused.bias),
                                       self.bn_reference(x), rtol=1e-9, atol=1e-9)

    def test_fused_forward_is_conv_then_bn(self):
        (fused,) = fuse_layers([self.conv, self.bn])
        for x in self.data:
            np.testing.assert_allclose(fused.forward(x), self.bn_reference(x),
                                       rtol=1e-9, atol=1e-9)

    def test_calibration_observes_the_output_after_bn(self):
        fused = fuse_layers([self.conv, self.bn])
        calibrate(fused, "x", MinMaxObserver(), self.data)
        ys = np.concatenate([self.bn_reference(x).ravel() for x in self.data])
        lo, hi = fused[0].observer.range()
        self.assertAlmostEqual(lo, float(ys.min()), places=5)
        self.assertAlmostEqual(hi, float(ys.max()), places=5)


IN, HID, T = 3, 5, 7


def lstm_state(seed=3):
    """nn.LSTM(IN, HID) state dict as numpy, gates i, f, g, o."""
    rng = np.random.default_rng(seed)
    return {
        "weight_ih_l0": rng.uniform(-1.0, 1.0, size=(4 * HID, IN)).astype(np.float32),
        "weight_hh_l0": rng.uniform(-0.6, 0.6, size=(4 * HID, HID)).astype(np.float32),
        "bias_ih_l0": rng.uniform(-0.5, 0.5, size=4 * HID).astype(np.float32),
        "bias_hh_l0": rng.uniform(-0.5, 0.5, size=4 * HID).astype(np.float32),
    }


def independent_lstm(state, seq):
    """nn.LSTM's equations in float64, written apart from the importer."""
    w_ih = state["weight_ih_l0"].astype(np.float64)
    w_hh = state["weight_hh_l0"].astype(np.float64)
    b = state["bias_ih_l0"].astype(np.float64) + state["bias_hh_l0"].astype(np.float64)
    h = np.zeros(HID)
    c = np.zeros(HID)
    hs, cs = [], []
    for x in np.asarray(seq, np.float64):
        i, f, g, o = np.split(w_ih @ x + w_hh @ h + b, 4)
        sig = lambda v: 1.0 / (1.0 + np.exp(-v))  # noqa: E731
        c = sig(f) * c + sig(i) * np.tanh(g)
        h = sig(o) * np.tanh(c)
        hs.append(h.copy())
        cs.append(c.copy())
    return hs, cs


def sequences(n, seed=5):
    rng = np.random.default_rng(seed)
    return [rng.uniform(-1.5, 1.5, size=(T, IN)).astype(np.float32) for _ in range(n)]


class LstmImport(unittest.TestCase):
    def make(self, cell_bits=16, state=None, **kwargs):
        return LSTM.from_state_dict("lstm", state or lstm_state(), input_name="x",
                                    sequence_length=T, cell_bits=cell_bits, **kwargs)

    def test_float_forward_matches_lstm_equations(self):
        state = lstm_state()
        lstm = self.make(state=state)
        for seq in sequences(4):
            hs, cs = lstm_float_forward(lstm, seq)
            ref_h, ref_c = independent_lstm(state, seq)
            np.testing.assert_allclose(np.stack(hs), np.stack(ref_h), atol=2e-6)
            np.testing.assert_allclose(np.stack(cs), np.stack(ref_c), atol=2e-6)

    def test_calibration_scales_hidden_and_cell_symmetrically(self):
        state = lstm_state()
        data = sequences(6)
        all_h, all_c = [], []
        for seq in data:
            hs, cs = independent_lstm(state, seq)
            all_h += hs
            all_c += cs
        h_absmax = float(np.max(np.abs(np.stack(all_h))))
        c_absmax = float(np.max(np.abs(np.stack(all_c))))
        for bits, qmax in ((16, 32767), (8, 127)):
            lstm = self.make(cell_bits=bits, state=state)
            ranges = calibrate([lstm], "x", MinMaxObserver(), data)
            scale, zp = ranges["lstm"]
            self.assertEqual(zp, 0)
            self.assertAlmostEqual(scale, h_absmax / 127.0, delta=1e-7)
            self.assertAlmostEqual(lstm.cell_scale, c_absmax / qmax,
                                   delta=1e-6 / qmax)
            self.assertEqual((lstm.cell_qmin, lstm.cell_qmax), (-qmax - 1, qmax))

    def test_weights_are_symmetric_per_gate_and_gate_major(self):
        state = lstm_state()
        lstm = self.make(state=state)
        ranges = calibrate([lstm], "x", MinMaxObserver(), sequences(2))
        quantize_weights([lstm], ranges)
        for name, w, scales, q in (
                ("input", state["weight_ih_l0"], lstm.input_weight_scales, lstm.q_weight_ih),
                ("recurrent", state["weight_hh_l0"], lstm.recurrent_weight_scales,
                 lstm.q_weight_hh)):
            blocks = w.reshape(4, HID, -1)
            self.assertEqual(q.shape, blocks.shape, name)
            for g in range(4):
                expect = float(np.max(np.abs(blocks[g]))) / 127.0
                self.assertAlmostEqual(scales[g], expect, delta=1e-9, msg=name)
                self.assertEqual(int(np.max(np.abs(q[g]))), 127, name)
                np.testing.assert_array_equal(
                    q[g], np.clip(np.vectorize(lround)(blocks[g] / scales[g]),
                                  -127, 127), err_msg=name)
        # The gates' scales differ, so one shared scale would be wrong.
        self.assertGreater(len(set(lstm.input_weight_scales)), 1)

    def test_biases_sum_both_vectors_in_the_lut_grid(self):
        state = lstm_state()
        lstm = self.make(state=state)
        ranges = calibrate([lstm], "x", MinMaxObserver(), sequences(2))
        quantize_weights([lstm], ranges)
        lut = float(np.float32(LSTM_LUT_INPUT_SCALE))
        b = (state["bias_ih_l0"].astype(np.float64)
             + state["bias_hh_l0"].astype(np.float64)).astype(np.float32)
        expect = [lround(float(v) / lut) for v in b]
        self.assertEqual(lstm.q_bias.dtype, np.int32)
        self.assertEqual(lstm.q_bias.shape, (4, HID))
        self.assertEqual(lstm.q_bias.reshape(-1).tolist(), expect)

    def test_bias_rounds_half_away_from_zero(self):
        lut = float(np.float32(LSTM_LUT_INPUT_SCALE))
        state = lstm_state()
        state["bias_ih_l0"] = np.zeros(4 * HID, np.float32)
        state["bias_hh_l0"] = np.zeros(4 * HID, np.float32)
        state["bias_ih_l0"][0] = np.float32(2.5 * lut)
        state["bias_ih_l0"][1] = np.float32(-2.5 * lut)
        lstm = self.make(state=state)
        quantize_weights([lstm], calibrate([lstm], "x", MinMaxObserver(), sequences(1)))
        self.assertEqual(lstm.q_bias.reshape(-1)[:2].tolist(),
                         [lround(float(np.float32(2.5 * lut)) / lut),
                          lround(float(np.float32(-2.5 * lut)) / lut)])
        self.assertEqual(abs(int(lstm.q_bias.reshape(-1)[0])),
                         abs(int(lstm.q_bias.reshape(-1)[1])))

    def test_head_reads_the_last_hidden_state(self):
        state = lstm_state()
        w = np.eye(2, HID, dtype=np.float32)
        seen = []

        def head_forward(h):
            seen.append(np.array(h))
            return h[:2]
        lstm = self.make(state=state)
        head = Dense(name="head", weight=w, input_name="lstm", forward=head_forward)
        data = sequences(3)
        calibrate([lstm, head], "x", MinMaxObserver(), data)
        for got, seq in zip(seen, data):
            np.testing.assert_allclose(got, independent_lstm(state, seq)[0][-1], atol=2e-6)

    def test_emitted_header_carries_every_cell_input(self):
        lstm = self.make()
        head = Dense(name="head", weight=np.ones((2, HID), np.float32),
                     bias=np.zeros(2, np.float32), input_name="lstm",
                     forward=lambda h: np.ones((2, HID)) @ h)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "w.hpp")
            import_pytorch_model([lstm, head], "x", MinMaxObserver(), sequences(3),
                                 path, "ns")
            with open(path) as f:
                text = f.read()
        for needle in ("klstm_Scale", "klstm_ZeroPoint = 0", "typedef int16_t lstm_CellType;",
                       "klstm_CellScale", "klstm_CellQMin      = -32768",
                       "klstm_CellQMax      = 32767", "klstm_LutInputScale",
                       f"klstm_NumInputs = {IN}", f"klstm_NumHidden = {HID}",
                       f"klstm_SequenceLength = {T}",
                       "constexpr float klstm_InputWeightScales[4]",
                       "constexpr float klstm_RecurrentWeightScales[4]",
                       f"constexpr int8_t klstm_InputWeights[{4 * HID * IN}]",
                       f"constexpr int8_t klstm_RecurrentWeights[{4 * HID * HID}]",
                       f"constexpr int32_t klstm_Biases[{4 * HID}]",
                       "khead_WeightScale"):
            self.assertIn(needle, text)

    def test_rejects_what_qlstmcell_cannot_run(self):
        state = lstm_state()
        for extra, why in (("weight_ih_l0_reverse", "bidirectional"),
                           ("weight_ih_l1", "stacked"),
                           ("weight_hr_l0", "proj_size")):
            bad = dict(state)
            bad[extra] = np.zeros(1, np.float32)
            with self.assertRaisesRegex(ValueError, why):
                LSTM.from_state_dict("lstm", bad, input_name="x")
        with self.assertRaisesRegex(ValueError, "cell_bits"):
            self.make(cell_bits=32)
        with self.assertRaisesRegex(ValueError, "weight_ih"):
            LSTM(name="lstm", input_name="x", weight_ih=np.zeros((4 * HID + 1, IN)),
                 weight_hh=np.zeros((4 * HID, HID)))
        with self.assertRaisesRegex(ValueError, "weight_hh"):
            LSTM(name="lstm", input_name="x", weight_ih=np.zeros((4 * HID, IN)),
                 weight_hh=np.zeros((4 * HID, HID + 1)))
        with self.assertRaisesRegex(ValueError, "sequence_length"):
            calibrate([self.make()], "x", MinMaxObserver(),
                      [np.zeros((T + 1, IN), np.float32)])

    def test_lstm_must_be_the_first_layer(self):
        pre = Dense(name="pre", weight=np.eye(IN, dtype=np.float32), input_name="x",
                    forward=lambda v: v)
        lstm = LSTM.from_state_dict("lstm", lstm_state(), input_name="pre")
        with self.assertRaisesRegex(ValueError, "first layer"):
            calibrate([pre, lstm], "x", MinMaxObserver(), sequences(1))

    @unittest.skipIf(torch is None, "needs torch")
    def test_float_forward_matches_torch_nn_lstm(self):
        torch.manual_seed(0)
        module = torch.nn.LSTM(IN, HID, batch_first=True)
        with torch.no_grad():
            for p in module.parameters():
                p.mul_(3.0)  # drive the gates into saturation as well
        state = {k: v.detach().numpy() for k, v in module.state_dict().items()}
        lstm = LSTM.from_state_dict("lstm", state, input_name="x")
        for seq in sequences(4):
            with torch.no_grad():
                out, (h_n, c_n) = module(torch.from_numpy(seq)[None])
            hs, cs = lstm_float_forward(lstm, seq)
            np.testing.assert_allclose(np.stack(hs), out[0].numpy(), atol=1e-5)
            np.testing.assert_allclose(cs[-1], c_n[0, 0].numpy(), atol=1e-5)
        for kwargs, why in (({"num_layers": 2}, "stacked"),
                            ({"bidirectional": True}, "bidirectional"),
                            ({"proj_size": 2}, "proj_size")):
            m = torch.nn.LSTM(IN, HID, **kwargs)
            st = {k: v.detach().numpy() for k, v in m.state_dict().items()}
            with self.assertRaisesRegex(ValueError, why):
                LSTM.from_state_dict("lstm", st, input_name="x")


def conv2d_nhwc(x, w, b, stride=(1, 1), padding=(0, 0, 0, 0)):
    """NHWC input, OHWI weights, zero padding (top, bottom, left, right)."""
    pt, pb, pl, pr = padding
    xp = np.pad(x, ((pt, pb), (pl, pr), (0, 0)))
    f, kh, kw, _ = w.shape
    oh = (xp.shape[0] - kh) // stride[0] + 1
    ow = (xp.shape[1] - kw) // stride[1] + 1
    y = np.empty((oh, ow, f), dtype=np.float64)
    for i in range(oh):
        for j in range(ow):
            patch = xp[i * stride[0]:i * stride[0] + kh, j * stride[1]:j * stride[1] + kw, :]
            y[i, j, :] = np.tensordot(w, patch, axes=([1, 2, 3], [0, 1, 2])) + b
    return y


REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

MODEL_DRIVER = """
#include <cstdio>
#include "@HEADER@"
int main()
{
    static @ACT@ input[@NS@::kModelInputSize];
    static @ACT@ output[@NS@::kModelOutputSize];
    alignas(4) static @ACT@ arena[@NS@::kModelArenaSize / sizeof(@ACT@)];
    for (;;)
    {
        for (std::size_t i = 0; i < @NS@::kModelInputSize; ++i)
        {
            int v;
            if (std::scanf("%d", &v) != 1)
            {
                return 0;
            }
            input[i] = static_cast<@ACT@>(v);
        }
        @NS@::forward(input, output, arena);
        for (std::size_t i = 0; i < @NS@::kModelOutputSize; ++i)
        {
            std::printf("%d ", static_cast<int>(output[i]));
        }
        std::printf("\\n");
    }
}
"""


class ModelCodegen(unittest.TestCase):
    """generate_model: the generated model.hpp, compiled freestanding with
    g++, against the float model it was calibrated on."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def build_and_run(self, layers, data, tag, inputs, **codegen):
        weights = os.path.join(self.dir, f"{tag}_weights.hpp")
        model = os.path.join(self.dir, f"{tag}_model.hpp")
        records, gen = generate_model(layers, "x", MinMaxObserver(), data, weights, model, tag, **codegen)
        wide = codegen.get("activation_bits", 8) == 16
        src = os.path.join(self.dir, f"{tag}.cpp")
        with open(src, "w") as f:
            f.write(MODEL_DRIVER.replace("@HEADER@", os.path.basename(model)).replace("@NS@", tag)
                    .replace("@ACT@", "int16_t" if wide else "int8_t"))
        exe = os.path.join(self.dir, tag)
        subprocess.run(["g++", "-std=c++14", "-O1", "-Wall", "-Wextra", "-Werror", "-Wpedantic",
                        "-DTINYMIND_ENABLE_QUANTIZATION=1", "-DTINYMIND_ENABLE_FLOAT=0", "-DTINYMIND_ENABLE_STD=0",
                        "-I" + self.dir, "-I" + os.path.join(REPO, "cpp"), "-I" + os.path.join(REPO, "cpp", "include"),
                        "-I" + os.path.join(REPO, "include"), src, "-o", exe], check=True)
        first = records[0]
        if wide:
            from tinymind_import import _onnx_tools
            _, cg = _onnx_tools()
            scale, zp, lo, hi = cg.int16_grid(first.input_scale, first.input_zp).scale, 0, -32768, 32767
        else:
            scale, zp, lo, hi = first.input_scale, first.input_zp, -128, 127
        q = [np.clip(np.round(x.reshape(-1) / scale) + zp, lo, hi).astype(int) for x in inputs]
        stdin = "\n".join(" ".join(str(v) for v in row) for row in q) + "\n"
        out = subprocess.run([exe], input=stdin, capture_output=True, text=True, check=True).stdout
        return records, gen, np.array([[int(v) for v in line.split()] for line in out.strip().splitlines()])

    def mlp(self):
        rng = np.random.default_rng(11)
        w1, b1 = rng.normal(0, 0.6, (16, 4)), rng.normal(0, 0.1, 16)
        w2, b2 = rng.normal(0, 0.4, (8, 16)), rng.normal(0, 0.1, 8)
        w3, b3 = rng.normal(0, 0.5, (3, 8)), rng.normal(0, 0.1, 3)
        layers = [Dense(name="fc1", weight=w1, bias=b1, input_name="x", forward=lambda x: x @ w1.T + b1),
                  ReLU(name="h1", input_name="fc1"),
                  Dense(name="fc2", weight=w2, bias=b2, input_name="h1", forward=lambda x: x @ w2.T + b2),
                  ReLU(name="h2", input_name="fc2"),
                  Dense(name="fc3", weight=w3, bias=b3, input_name="h2", forward=lambda x: x @ w3.T + b3),
                  Softmax(name="probs", input_name="fc3")]

        def reference(x):
            h = np.maximum(x @ w1.T + b1, 0)
            h = np.maximum(h @ w2.T + b2, 0)
            z = h @ w3.T + b3
            e = np.exp(z - z.max())
            return e / e.sum()
        data = [rng.normal(0, 1, 4) for _ in range(256)]
        test = [rng.normal(0, 1, 4) for _ in range(200)]
        return layers, reference, data, test

    def test_mlp_folds_each_relu_and_tracks_float(self):
        layers, reference, data, test = self.mlp()
        records, gen, out = self.build_and_run(layers, data, "mlp", test)
        self.assertEqual([r.op_type for r in records], ["Dense", "Dense", "Dense", "Softmax"])
        self.assertEqual([r.output_zp for r in records[:2]], [-128, -128])  # the folded ReLUs
        want = np.array([reference(x) for x in test])
        got = (out + 128) / 256.0
        err = np.abs(got - want)
        agree = np.mean(got.argmax(1) == want.argmax(1))
        print(f"\n  generated MLP: mean |p - float| {err.mean():.4f}, max {err.max():.4f}, argmax {agree:.3f}")
        self.assertLess(err.mean(), 0.02)
        self.assertGreater(agree, 0.95)

        # 16-bit activations (TFLite's 16x8 scheme, its softmax included),
        # against the same int8 weights run with float activations: the
        # weights' own rounding is common to both models.
        from tinymind_import import _onnx_tools
        _, cg = _onnx_tools()
        _, _, out16 = self.build_and_run(layers, data, "mlp16", test, activation_bits=16)
        ref = np.array([cg.float_reference(records, x) for x in test])
        e8 = np.abs(got - ref)
        e16 = np.abs(out16 / 32768.0 - ref)
        print(f"  int8 weights with float activations: int8 model mean {e8.mean():.5f}, "
              f"16x8 model mean {e16.mean():.6f}")
        self.assertLess(20 * e16.mean(), e8.mean())

    def cnn(self):
        rng = np.random.default_rng(12)
        w1, b1 = rng.normal(0, 0.5, (4, 3, 3, 1)), rng.normal(0, 0.1, 4)
        w2, b2 = rng.normal(0, 0.3, (6, 3, 3, 4)), rng.normal(0, 0.1, 6)
        w3, b3 = rng.normal(0, 0.2, (5, 4 * 4 * 6)), rng.normal(0, 0.1, 5)
        gamma, beta = rng.uniform(0.5, 2.0, 4), rng.normal(0, 0.3, 4)
        mean, var = rng.normal(0, 0.2, 4), rng.uniform(0.5, 1.5, 4)
        sigma = gamma / np.sqrt(var + 1e-5)
        pad1, pad2 = (1, 1, 1, 1), (0, 1, 0, 1)
        layers = [Conv2D(name="c1", weight=w1, bias=b1, input_name="x", input_shape=(8, 8, 1), padding=pad1,
                         forward=lambda x: conv2d_nhwc(x, w1, b1, padding=pad1)),
                  BatchNorm2D(name="bn", input_name="c1", gamma=gamma, beta=beta, mean=mean, variance=var),
                  ReLU(name="r1", input_name="bn"),
                  Conv2D(name="c2", weight=w2, bias=b2, input_name="r1", input_shape=(8, 8, 4), stride=(2, 2),
                         padding=pad2, forward=lambda x: conv2d_nhwc(x, w2, b2, (2, 2), pad2)),
                  ReLU(name="r2", input_name="c2"),
                  Dense(name="fc", weight=w3, bias=b3, input_name="r2",
                        forward=lambda x: x.reshape(-1) @ w3.T + b3)]

        def reference(x):
            h = np.maximum((conv2d_nhwc(x, w1, b1, padding=pad1) - mean) * sigma + beta, 0)
            h = np.maximum(conv2d_nhwc(h, w2, b2, (2, 2), pad2), 0)
            return h.reshape(-1) @ w3.T + b3
        data = [rng.normal(0, 1, (8, 8, 1)) for _ in range(128)]
        test = [rng.normal(0, 1, (8, 8, 1)) for _ in range(100)]
        return layers, reference, data, test

    def test_cnn_with_batchnorm_padding_and_stride_tracks_float(self):
        layers, reference, data, test = self.cnn()
        records, gen, out = self.build_and_run(layers, data, "cnn", test)
        self.assertEqual([r.op_type for r in records], ["Conv", "Conv", "Dense"])
        self.assertEqual(records[1].geometry["OutputHeight"], 4)
        last = records[-1]
        want = np.array([reference(x) for x in test])
        got = (out - last.output_zp) * last.output_scale
        span = 255 * last.output_scale
        err = np.abs(got - want) / span
        print(f"\n  generated CNN: mean |y - float| {err.mean():.4f} of the output range, max {err.max():.4f}")
        self.assertLess(err.mean(), 0.01)
        self.assertLess(err.max(), 0.05)

        # Every weight layer on int4: further from float, still close.
        _, gen4, out4 = self.build_and_run(layers, data, "cnn4", test, int4_dense=True, int4_conv=True)
        err4 = np.abs((out4 - last.output_zp) * last.output_scale - want) / span
        print(f"  generated CNN, int4 weights: mean {err4.mean():.4f}, weights {gen4.weight_bytes} B "
              f"against {gen.weight_bytes} B")
        self.assertLess(err4.mean(), 0.05)
        self.assertLess(gen4.weight_bytes, gen.weight_bytes * 0.55)

    def test_refuses_what_it_cannot_generate(self):
        layers, _, data, _ = self.cnn()
        layers[0].input_shape = None
        with self.assertRaisesRegex(ValueError, "input_shape"):
            generate_model(layers, "x", MinMaxObserver(), data[:4], os.path.join(self.dir, "w.hpp"),
                           os.path.join(self.dir, "m.hpp"), "net")
        rng = np.random.default_rng(13)
        moe = MoE(name="moe", input_name="x", router_weight=rng.normal(0, 1, (2, 4)),
                  expert_weights=[rng.normal(0, 1, (3, 4)) for _ in range(2)])
        with self.assertRaisesRegex(ValueError, "no whole-model code generation for MoE"):
            generate_model([moe], "x", MinMaxObserver(), [rng.normal(0, 1, 4) for _ in range(4)],
                           os.path.join(self.dir, "w.hpp"), os.path.join(self.dir, "m.hpp"), "net")


if __name__ == "__main__":
    unittest.main()
