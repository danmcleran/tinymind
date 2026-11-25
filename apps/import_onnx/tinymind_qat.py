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

"""Quantization-aware training (QAT) for tinymind's int4 weights.

The code generator's int4 options (tinymind_model_codegen.py: int4_dense,
int4_conv) requantize a trained model's weights to symmetric per-channel
int4 after training: scale max |w| / 7 per output channel, codes
round(w / scale) in -7..7, ties away from zero (requantize_int4, and
quantizeInt4PerChannel in cpp/qint4.hpp). That costs accuracy the
training never saw. prepare_int4_qat() puts exactly that projection on a
PyTorch model's Conv2d and Linear weights, with a straight-through
gradient, so fine-tuning learns weights that survive it:

    import tinymind_qat
    tinymind_qat.prepare_int4_qat(model)
    ...fine-tune a few epochs at a small learning rate...
    tinymind_qat.finalize_int4_qat(model)   # bakes the int4-grid weights in
    torch.onnx.export(model, ...)           # then quantize and generate as usual

The finalized weights lie on each channel's int4 grid, and that grid passes
through ONNX Runtime's per-channel int8 quantization and the code
generator's int4 requantization unchanged: the int8 scale is 7 / 127 of the
int4 one, every int4 code k becomes the int8 code round(18.14 k), and
requantizing to int4 rounds that back to k. int4_grid_codes() checks it.

Needs torch; never imported by the code generator itself.
"""

import torch
import torch.nn as nn
from torch.nn.utils import parametrize


def int4_per_channel(w: torch.Tensor) -> tuple:
    """(codes, scales) of w on the code generator's int4 grid: per output
    channel (dimension 0), scale max |w| / 7 (1 for an all-zero channel),
    codes round(w / scale) clamped to -7..7, ties away from zero."""
    flat = w.detach().reshape(w.shape[0], -1)
    largest = flat.abs().amax(dim=1)
    scales = torch.where(largest > 0, largest / 7.0, torch.ones_like(largest))
    x = flat / scales[:, None]
    codes = torch.clamp(torch.sign(x) * torch.floor(x.abs() + 0.5), -7, 7)
    return codes.reshape(w.shape), scales


class Int4Weight(nn.Module):
    """A parametrization: the weight as the int4 grid holds it in the forward
    pass; gradients pass straight through to the float weight."""

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        codes, scales = int4_per_channel(w)
        shape = (-1,) + (1,) * (w.dim() - 1)
        projected = codes * scales.reshape(shape)
        return w + (projected - w).detach()


def int4_modules(model: nn.Module, types=(nn.Conv2d, nn.Linear)) -> list:
    return [m for m in model.modules() if isinstance(m, types)]


def prepare_int4_qat(model: nn.Module, types=(nn.Conv2d, nn.Linear)) -> nn.Module:
    """Fake-quantize the weight of every Conv2d and Linear (or `types`) in
    model to int4 per output channel, in place. Returns model."""
    for m in int4_modules(model, types):
        parametrize.register_parametrization(m, "weight", Int4Weight())
    return model


def finalize_int4_qat(model: nn.Module) -> nn.Module:
    """Replace each fake-quantized weight with its int4-grid value, so the
    model exports as ordinary float weights. Returns model."""
    for m in model.modules():
        if parametrize.is_parametrized(m, "weight"):
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
    return model


def int4_grid_codes(w: torch.Tensor) -> bool:
    """True when every channel of w already lies on its int4 grid."""
    codes, scales = int4_per_channel(w)
    shape = (-1,) + (1,) * (w.dim() - 1)
    return bool(torch.allclose(codes * scales.reshape(shape), w.detach(), rtol=0, atol=1e-6 * float(scales.max())))
