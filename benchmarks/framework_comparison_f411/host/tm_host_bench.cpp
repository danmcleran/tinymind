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

// Host timing: the same generated TinyMind int8 models the firmware runs,
// timed with std::chrono on the build host. Built once per SIMD backend by
// `make host-bench`, so the x86 vector paths (AVX2, AVX-VNNI, AVX-512F,
// AVX-512 VNNI) can be compared against the scalar build. Prints one
// key=value line: whole-model ns per inference for the MLP and the KWS CNN,
// then ns per call for each KWS layer. Exit status is non-zero if the outputs
// do not reproduce the expected hashes (fw/backend.hpp), so a backend that is
// fast but wrong cannot report a number.
#define MLP_DATA_INT8
#define KWS_DATA_INT8
#include "mlp_data.h"
#include "kws_data.h"
#include "mlp_tinymind.hpp"
#include "kws_tinymind.hpp"
#include "backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

namespace
{

constexpr int kReps = 41;       // timed repetitions; the median is reported
constexpr int kLoops = 200;     // inferences or layer calls per repetition
constexpr int kMlpPasses = 20;  // passes over the MLP test set per repetition

volatile int sink;

uint32_t fnv(uint32_t h, const int8_t* p, int n)
{
    for (int i = 0; i < n; ++i)
    {
        h ^= static_cast<uint8_t>(p[i]);
        h *= 0x01000193u;
    }
    return h;
}

// Median ns per unit of work over kReps runs of f, where one run does perRep units.
template<typename F>
double medianNs(F f, int perRep)
{
    std::vector<double> samples;
    samples.reserve(kReps);
    f();
    for (int r = 0; r < kReps; ++r)
    {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / perRep);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

// Out of line so each layer is timed as its own call, as on the board.
template<typename Layer, typename In, typename Out>
__attribute__((noinline)) void runLayer(const Layer& layer, const In* in, Out* out)
{
    layer.forward(in, out);
}

int8_t c1[tm_kws::C1Type::OutputSize];
int8_t p1[tm_kws::P1Type::OutputSize];
int8_t dw[tm_kws::DwType::OutputSize];
int8_t pw[tm_kws::PwType::OutputSize];
int8_t gap[tm_kws::GapType::OutputSize];
int8_t logits[KWS_N_OUT];

} // namespace

int main()
{
    uint32_t mlpHash = 0x811C9DC5u;
    for (int s = 0; s < MLP_N_TEST; ++s)
    {
        int8_t out[MLP_N_OUT];
        tm_mlp::forward(mlp_test_x_i8 + s * MLP_N_IN, out);
        mlpHash = fnv(mlpHash, out, MLP_N_OUT);
    }
    uint32_t kwsHash = 0x811C9DC5u;
    for (int s = 0; s < KWS_N_TEST; ++s)
    {
        int8_t out[KWS_N_OUT];
        tm_kws::forward(kws_test_x_i8 + s * KWS_N_IN, out);
        kwsHash = fnv(kwsHash, out, KWS_N_OUT);
    }
    const bool hashOk = (mlpHash == TM_MLP_HASH) && (kwsHash == TM_KWS_HASH);

    const double mlp = medianNs([] {
        for (int pass = 0; pass < kMlpPasses; ++pass)
        {
            for (int s = 0; s < MLP_N_TEST; ++s)
            {
                int8_t out[MLP_N_OUT];
                tm_mlp::forward(mlp_test_x_i8 + s * MLP_N_IN, out);
                sink = out[0];
            }
        }
    }, kMlpPasses * MLP_N_TEST);
    const double kws = medianNs([] {
        for (int l = 0; l < kLoops; ++l)
        {
            tm_kws::forward(kws_test_x_i8 + (l % KWS_N_TEST) * KWS_N_IN, logits);
            sink = logits[0];
        }
    }, kLoops);

    // Fill every intermediate from the first test sample, then time each layer on it.
    const int8_t* in = kws_test_x_i8;
    runLayer(tm_kws::C1, in, c1);
    runLayer(tm_kws::P1, c1, p1);
    runLayer(tm_kws::DW, p1, dw);
    runLayer(tm_kws::PW, dw, pw);
    runLayer(tm_kws::GAP, pw, gap);
    const double conv = medianNs([in] { for (int l = 0; l < kLoops; ++l) { runLayer(tm_kws::C1, in, c1); } }, kLoops);
    const double maxpool = medianNs([] { for (int l = 0; l < kLoops; ++l) { runLayer(tm_kws::P1, c1, p1); } }, kLoops);
    const double depthwise = medianNs([] { for (int l = 0; l < kLoops; ++l) { runLayer(tm_kws::DW, p1, dw); } }, kLoops);
    const double pointwise = medianNs([] { for (int l = 0; l < kLoops; ++l) { runLayer(tm_kws::PW, dw, pw); } }, kLoops);
    const double avgpool = medianNs([] { for (int l = 0; l < kLoops; ++l) { runLayer(tm_kws::GAP, pw, gap); } }, kLoops);
    const double fc = medianNs([] { for (int l = 0; l < kLoops; ++l) { runLayer(tm_kws::D, gap, logits); } }, kLoops);

    std::printf("mlp_ns=%.0f kws_ns=%.0f conv_ns=%.0f maxpool_ns=%.0f depthwise_ns=%.0f "
                "pointwise_ns=%.0f avgpool_ns=%.0f fc_ns=%.0f hash_ok=%d\n",
                mlp, kws, conv, maxpool, depthwise, pointwise, avgpool, fc, hashOk ? 1 : 0);
    if (!hashOk)
    {
        std::fprintf(stderr, "hash mismatch: mlp 0x%08X (expect 0x%08X), kws 0x%08X (expect 0x%08X)\n",
                     mlpHash, TM_MLP_HASH, kwsHash, TM_KWS_HASH);
        return 1;
    }
    return 0;
}
