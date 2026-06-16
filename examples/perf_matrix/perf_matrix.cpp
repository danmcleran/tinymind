/**
* Copyright (c) 2026 Dan McLeran
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

// Phase 14 perf matrix example.
//
// Runs a fixed MobileNetV2-shaped int8 block (QPointwiseConv2D ->
// QDepthwiseConv2D -> QPointwiseConv2D -> QDense) with whichever SIMD
// gates were set at build time, and emits a single CSV row with:
//
//   active_backend, qconv_iters, qconv_us, qdense_iters, qdense_us,
//   output_checksum_int64
//
// To compare backends, build the binary multiple times with different
// TINYMIND_ENABLE_SIMD_* flags and concatenate the CSV rows. The
// output_checksum is invariant across backends (Phase 14 bit-exactness
// invariant): if it changes between two backends the corresponding gate
// is broken.
//
// All weights / biases / activations are deterministic (seeded
// sawtooth fills), so the checksum is reproducible.

#include "qaffine.hpp"
#include "qdense.hpp"
#include "qconv2d.hpp"
#include "qdepthwiseconv2d.hpp"
#include "qpointwiseconv2d.hpp"
#include "qactivations.hpp"
#include "qpool2d.hpp"
#include "include/simd/simd_dispatch.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace {

using tinymind::QPointwiseConv2D;
using tinymind::QDepthwiseConv2D;
using tinymind::QDense;
using tinymind::Requantizer;
using tinymind::qreluBuffer;
using tinymind::QGlobalAvgPool2D;

constexpr std::size_t H  = 14;
constexpr std::size_t W  = 14;
constexpr std::size_t IC = 32;   // expansion-stage channels
constexpr std::size_t MC = 32;   // intermediate channels after pw1
constexpr std::size_t OC = 64;   // output channels of the block
constexpr std::size_t KH = 3;
constexpr std::size_t KW = 3;
constexpr std::size_t NumClasses = 10;

typedef QPointwiseConv2D<int8_t, int8_t, int32_t, int8_t, H, W, IC, MC>     PW1Type;
typedef QDepthwiseConv2D<int8_t, int8_t, int32_t, int8_t,
                         H - KH + 1, W - KW + 1, MC, KH, KW, 1, 1>           DwType;
// Pointwise after a depthwise that consumed the 3x3 spatial window.
typedef QPointwiseConv2D<int8_t, int8_t, int32_t, int8_t,
                         DwType::OutputHeight, DwType::OutputWidth, MC, OC>  PW2Type;
typedef QGlobalAvgPool2D<int8_t, int32_t,
                         PW2Type::OutputHeight, PW2Type::OutputWidth, OC>    GapType;
typedef QDense<int8_t, int8_t, int32_t, int8_t, OC, NumClasses>              DenseType;

// We do not need a 3x3 conv ahead of the depthwise — the pointwise
// expansion above is followed directly by depthwise on the same
// spatial extent, then a stride-1 depthwise reduces to (H-2)x(W-2).
// Conv2D path is exercised separately for the cycle bench so we get a
// proper kernel-loop number, not just a 1x1 multiply-add.
#include "qconv2d.hpp"
typedef tinymind::QConv2D<int8_t, int8_t, int32_t, int8_t, H, W, IC, KH, KW,
                          1, 1, OC> ConvType;

// Weight + activation buffers.
int8_t  conv_w[ConvType::TotalWeights];
int32_t conv_b[ConvType::NumFilters];

int8_t  pw1_w[PW1Type::TotalWeights];
int32_t pw1_b[PW1Type::NumFilters];

int8_t  dw_w[DwType::TotalWeights];
int32_t dw_b[DwType::Channels];
Requantizer<int32_t, int8_t> dw_req[DwType::Channels];

int8_t  pw2_w[PW2Type::TotalWeights];
int32_t pw2_b[PW2Type::NumFilters];

int8_t  dense_w[DenseType::InputLength * DenseType::OutputLength];
int32_t dense_b[DenseType::OutputLength];

int8_t input_buf[ConvType::InputSize];

// Bench scratch buffers.
int8_t conv_out[ConvType::OutputSize];
int8_t pw1_out [PW1Type::OutputSize];
int8_t dw_out  [DwType::OutputSize];
int8_t pw2_out [PW2Type::OutputSize];
int8_t gap_out [GapType::OutputSize];
int8_t cls_out [DenseType::OutputLength];

void fillSawtooth(int8_t* p, std::size_t n, int seed)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        const int v = static_cast<int>((i * 11 + seed * 13) & 0xFF) - 128;
        p[i] = static_cast<int8_t>(v);
    }
}

void fillI32(int32_t* p, std::size_t n, int seed)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        p[i] = (static_cast<int32_t>(i) * 7 - static_cast<int32_t>(seed) * 5);
    }
}

Requantizer<int32_t, int8_t> defaultRequantizer()
{
    Requantizer<int32_t, int8_t> r;
    r.multiplier = static_cast<int32_t>(1) << 29;
    r.shift = 7;
    r.zero_point = 0;
    r.qmin = -128;
    r.qmax = 127;
    return r;
}

void initialize()
{
    fillSawtooth(input_buf, ConvType::InputSize, 1);

    fillSawtooth(conv_w, ConvType::TotalWeights, 17);
    fillI32     (conv_b, ConvType::NumFilters,    3);

    fillSawtooth(pw1_w, PW1Type::TotalWeights, 23);
    fillI32     (pw1_b, PW1Type::NumFilters,    5);

    fillSawtooth(dw_w, DwType::TotalWeights, 31);
    fillI32     (dw_b, DwType::Channels,      7);
    for (std::size_t c = 0; c < DwType::Channels; ++c)
    {
        dw_req[c] = defaultRequantizer();
    }

    fillSawtooth(pw2_w, PW2Type::TotalWeights, 41);
    fillI32     (pw2_b, PW2Type::NumFilters,   11);

    fillSawtooth(dense_w, DenseType::InputLength * DenseType::OutputLength, 47);
    fillI32     (dense_b, DenseType::OutputLength,                          13);
}

// Bench helpers: each runs N iterations of one layer and returns elapsed us.
template<typename Layer>
double benchPwOrConv(Layer& layer, const int8_t* in, int8_t* out, int iters)
{
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
    {
        layer.forward(in, out);
    }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

int64_t outputChecksum(const int8_t* p, std::size_t n)
{
    // Mix arithmetic runs in unsigned: signed overflow and left-shift of a
    // large signed value are undefined behavior. Unsigned wrap is well-defined
    // and bit-identical to the prior two's-complement result, so the checksum
    // value is unchanged.
    uint64_t sum = 0;
    uint64_t mix = 1469598103934665603ULL;
    for (std::size_t i = 0; i < n; ++i)
    {
        const uint64_t v = static_cast<uint64_t>(static_cast<int64_t>(p[i]));
        sum += v;
        mix ^= v + (mix << 5) + (mix >> 2);
    }
    return static_cast<int64_t>(sum ^ mix);
}

} // namespace

int main()
{
    initialize();

    ConvType conv;
    conv.weights = conv_w;
    conv.biases  = conv_b;
    conv.input_zero_point = static_cast<int8_t>(-5);
    conv.requantizer = defaultRequantizer();

    PW1Type pw1;
    pw1.weights = pw1_w;
    pw1.biases  = pw1_b;
    pw1.input_zero_point = static_cast<int8_t>(-3);
    pw1.requantizer = defaultRequantizer();

    DwType dw;
    dw.weights = dw_w;
    dw.biases  = dw_b;
    dw.input_zero_point = static_cast<int8_t>(0);
    dw.requantizers = dw_req;

    PW2Type pw2;
    pw2.weights = pw2_w;
    pw2.biases  = pw2_b;
    pw2.input_zero_point = static_cast<int8_t>(0);
    pw2.requantizer = defaultRequantizer();

    GapType gap;
    gap.qmin = -128;
    gap.qmax = 127;

    DenseType dense;
    dense.weights = dense_w;
    dense.biases  = dense_b;
    dense.input_zero_point = static_cast<int8_t>(0);
    dense.requantizer = defaultRequantizer();

    // Warm-up and capture one full forward pass for the checksum.
    pw1.forward(input_buf, pw1_out);
    qreluBuffer<int8_t>(pw1_out, PW1Type::OutputSize, static_cast<int8_t>(0));
    dw.forward(pw1_out, dw_out);
    qreluBuffer<int8_t>(dw_out, DwType::OutputSize, static_cast<int8_t>(0));
    pw2.forward(dw_out, pw2_out);
    qreluBuffer<int8_t>(pw2_out, PW2Type::OutputSize, static_cast<int8_t>(0));
    gap.forward(pw2_out, gap_out);
    dense.forward(gap_out, cls_out);
    const int64_t cls_checksum =
        outputChecksum(cls_out, DenseType::OutputLength);

    conv.forward(input_buf, conv_out);
    const int64_t conv_checksum =
        outputChecksum(conv_out, ConvType::OutputSize);

    // Bench: tight loops on the two hottest layers.
    const int conv_iters  = 20;
    const int dense_iters = 5000;

    const double conv_us  = benchPwOrConv(conv,  input_buf, conv_out, conv_iters);
    const double dense_us = benchPwOrConv(dense, gap_out,   cls_out,  dense_iters);

    std::printf("active_backend,conv_iters,conv_total_us,conv_us_per_call,"
                "dense_iters,dense_total_us,dense_us_per_call,"
                "conv_output_checksum,dense_output_checksum\n");
    std::printf("%s,%d,%.3f,%.3f,%d,%.3f,%.3f,%lld,%lld\n",
                tinymind::simd::activeBackendName(),
                conv_iters, conv_us, conv_us / conv_iters,
                dense_iters, dense_us, dense_us / dense_iters,
                static_cast<long long>(conv_checksum),
                static_cast<long long>(cls_checksum));
    return 0;
}
