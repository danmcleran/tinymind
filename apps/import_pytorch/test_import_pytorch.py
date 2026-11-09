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

import unittest

import numpy as np

from tinymind_import import (BatchNorm2D, Conv2D, MinMaxObserver, calibrate,
                             fuse_layers)


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


if __name__ == "__main__":
    unittest.main()
