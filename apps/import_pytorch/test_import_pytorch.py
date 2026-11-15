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
import tempfile
import unittest

import numpy as np

from tinymind_import import (LSTM, LSTM_LUT_INPUT_SCALE, BatchNorm2D, Conv2D,
                             Dense, MinMaxObserver, calibrate, fuse_layers,
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


if __name__ == "__main__":
    unittest.main()
