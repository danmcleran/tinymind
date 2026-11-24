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

// Board image: cycles per call for the work merged in #295 to #300. One
// workload group per image, so each fits the L031K6's 32 KB of flash and
// reports its own size:
//
//   WORK_INT8      the generated digits model, int8
//   WORK_INT4      the same model with its dense layer on packed int4 weights
//   WORK_16X8      the same model with int16 activations (16x8)
//   WORK_FEATURES  a 256-sample log-mel frame (32 bands) and its 13 MFCCs,
//                  k-means and Gaussian anomaly scores (32 features), and one
//                  k-means learning step
//
// Prints one key=value line per workload over the ST-LINK virtual COM port.
#include "board.hpp"
#include "tier1_common.hpp"
#include "tables.h"

#ifndef REPS
#define REPS 33
#endif

#if !defined(WORK_INT8) && !defined(WORK_INT4) && !defined(WORK_16X8) && !defined(WORK_FEATURES)
#error "define one of WORK_INT8, WORK_INT4, WORK_16X8, WORK_FEATURES"
#endif

namespace
{

uint32_t gT[REPS];
volatile int32_t gSink;

void sort(uint32_t* a, int n)
{
    for (int i = 1; i < n; ++i)
    {
        const uint32_t x = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > x)
        {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = x;
    }
}

void kv(const char* key, uint32_t value)
{
    board_puts(key);
    board_puts("=");
    board_put_u32(value);
    board_puts(" ");
}

// One line: the median, minimum and maximum of gT, the stack in use at the
// deepest point of one painted call, and whether the outputs matched.
void report(const char* name, uint32_t stack, bool match)
{
    sort(gT, REPS);
    board_puts("work=");
    board_puts(name);
    board_puts(" ");
    kv("med", gT[REPS / 2]);
    kv("min", gT[0]);
    kv("max", gT[REPS - 1]);
    kv("stack", stack);
    kv("match", match ? 1u : 0u);
    board_puts("\n");
}

uint32_t stackOf(void (*fn)())
{
    board_stack_paint();
    fn();
    return board_stack_used();
}

void none()
{
}

#if defined(WORK_INT8) || defined(WORK_INT4)
alignas(digits::kModelArenaAlignment) int8_t gArena[digits::kModelArenaSize];
int8_t gIn[t1::kFrames][digits::kModelInputSize];
int8_t gOut[digits::kModelOutputSize];
#endif
#if defined(WORK_INT8)
const char* const kName = "digits_int8";
void runOne(int s) { t1::runDigitsInt8(gIn[s], gOut, gArena); }
uint32_t hashOutputs() { return t1::hashDigitsInt8(kDigitPixels, gArena); }
const uint32_t kExpect = HASH_INT8;
#elif defined(WORK_INT4)
const char* const kName = "digits_int4";
void runOne(int s) { t1::runDigitsInt4(gIn[s], gOut, gArena); }
uint32_t hashOutputs() { return t1::hashDigitsInt4(kDigitPixels, gArena); }
const uint32_t kExpect = HASH_INT4;
#elif defined(WORK_16X8)
alignas(digits::int16x8::kModelArenaAlignment) int16_t gArena[digits::int16x8::kModelArenaSize / sizeof(int16_t)];
int16_t gIn[t1::kFrames][digits::kModelInputSize];
int16_t gOut[digits::kModelOutputSize];
const char* const kName = "digits_16x8";
void runOne(int s) { t1::runDigits16x8(gIn[s], gOut, gArena); }
uint32_t hashOutputs() { return t1::hashDigits16x8(kDigitPixels, gArena); }
const uint32_t kExpect = HASH_16X8;
#endif

#if !defined(WORK_FEATURES) && defined(TIME_LAYERS)
// The model's seven layers one at a time, in forward()'s order, each in its
// own noinline function: the first two outputs in the arena, the rest in
// gTail. Layer names and sizes are those of examples/onnx_digits_int8.
#if defined(WORK_INT8)
namespace mns = digits;
#elif defined(WORK_INT4)
namespace mns = digits::int4;
#else
namespace mns = digits::int16x8;
#endif
namespace md = mns::model_detail;
#if defined(WORK_16X8)
typedef int16_t Act;
#else
typedef int8_t Act;
#endif
constexpr int kLayers = 7;
const char* const kLayerNames[kLayers] = {"c1_conv", "c2_conv", "add", "maxpool", "dw_conv", "pw_conv", "fc"};
Act gTail[128 + 128 + 256];
Act gLayerOut[digits::kModelOutputSize];
uint32_t gLayerT[kLayers][REPS];
Act* const kA = reinterpret_cast<Act*>(gArena);
Act* const kB = reinterpret_cast<Act*>(gArena) + 512;
Act* const kC = gTail;
Act* const kD = gTail + 128;
Act* const kE = gTail + 256;

__attribute__((noinline)) void layer0(const Act* in) { md::c1_Conv_layer.forward(in, kA); }
__attribute__((noinline)) void layer1(const Act*) { md::c2_Conv_layer.forward(kA, kB); }
__attribute__((noinline)) void layer2(const Act*) { md::Add_layer.forward(kA, kB, kA); }
__attribute__((noinline)) void layer3(const Act*) { md::MaxPool_layer.forward(kA, kC); }
__attribute__((noinline)) void layer4(const Act*) { md::dw_Conv_layer.forward(kC, kD); }
__attribute__((noinline)) void layer5(const Act*) { md::pw_Conv_layer.forward(kD, kE); }
__attribute__((noinline)) void layer6(const Act*) { md::fc_Gemm_layer.forward(kE, gLayerOut); }
void (*const kLayerFns[kLayers])(const Act*) = {layer0, layer1, layer2, layer3, layer4, layer5, layer6};

// Time every layer over REPS inputs; true when the layered output equals
// forward()'s for every input.
bool timeLayers()
{
    static_assert(sizeof(mns::ModelActivationType) == sizeof(Act), "the generated model's activation type");
    bool same = true;
    for (int r = 0; r < REPS; ++r)
    {
        const int s = r % static_cast<int>(t1::kFrames);
        for (int l = 0; l < kLayers; ++l)
        {
            const uint32_t a = board_cycles();
            kLayerFns[l](gIn[s]);
            gLayerT[l][r] = board_cycles() - a;
        }
        runOne(s);
        for (std::size_t c = 0; c < digits::kModelOutputSize; ++c)
        {
            same = same && (gLayerOut[c] == gOut[c]);
        }
    }
    for (int l = 0; l < kLayers; ++l)
    {
        sort(gLayerT[l], REPS);
        board_puts("layer=");
        board_puts(kLayerNames[l]);
        board_puts(" ");
        kv("med", gLayerT[l][REPS / 2]);
        board_puts("\n");
    }
    return same;
}
#endif

#if defined(WORK_FEATURES)
tinymind::QMelWorkspace<t1::kMelN> gWs;
int16_t gFrame[t1::kMelN];
int16_t gLogMel[t1::kBands];
int32_t gCoeffs[t1::kCoeffs];
int8_t gFeat[t1::kFrames][t1::kFeatures];
t1::Trainer gTrainer;
const t1::Mel kMel = {kWindow, kTwCos, kTwSin, kBandStart, kBandLength, kMelWeights, t1::kPreemphasis, -32768};
const t1::Mfcc kMfcc = {kDct};
const t1::KMeans kKMeans = {kCentroids, kInvRadius, kFeatureWeight};
const t1::Gaussian kGauss = {kGaussMeans, kGaussPrecision, kGaussBias};

void oneMel() { t1::runMel(kMel, gFrame, gLogMel, gWs); }
void oneMfcc() { t1::runMfcc(kMfcc, gLogMel, gCoeffs); }
void oneKMeans() { gSink = t1::runKMeans(kKMeans, gFeat[0]); }
void oneGauss() { gSink = t1::runGaussian(kGauss, gFeat[0]); }
#else
void oneModel() { runOne(0); }
#endif

[[noreturn]] void halt()
{
    board_flush();
    for (;;)
    {
        // A side-effect-free infinite loop is undefined behaviour in C++.
        __asm volatile ("" ::: "memory");
    }
}

} // namespace

int main()
{
    board_init();
    board_puts("\n# begin bench\n");

    // Each workload: one untimed call (warms the flash accelerator and
    // branch state), then REPS timed calls cycling through the inputs.
#if !defined(WORK_FEATURES)
    for (uint32_t s = 0; s < t1::kFrames; ++s)
    {
#if defined(WORK_16X8)
        t1::digitsInput16(kDigitPixels[s], gIn[s]);
#else
        t1::digitsInput8(kDigitPixels[s], gIn[s]);
#endif
    }
    const bool match = hashOutputs() == kExpect;
    runOne(0);
    for (int r = 0; r < REPS; ++r)
    {
        const uint32_t a = board_cycles();
        runOne(r % static_cast<int>(t1::kFrames));
        gT[r] = board_cycles() - a;
    }
    report(kName, stackOf(oneModel), match);
#if defined(TIME_LAYERS)
    const bool layersMatch = timeLayers();
    kv("layers_match", layersMatch ? 1u : 0u);
    board_puts("\n");
#endif
#else
    for (uint32_t s = 0; s < t1::kFrames; ++s)
    {
        t1::makeFeatures(gFeat[s], s + 1);
    }
    uint32_t h[4];
    t1::hashFeatures(kMel, kMfcc, kKMeans, kGauss, gWs, gFrame, h);

    // The frame is regenerated outside the timed region.
    t1::makeFrame(gFrame, t1::kMelN, 1);
    oneMel();
    for (int r = 0; r < REPS; ++r)
    {
        t1::makeFrame(gFrame, t1::kMelN, 1u + static_cast<uint32_t>(r) % t1::kFrames);
        const uint32_t a = board_cycles();
        t1::runMel(kMel, gFrame, gLogMel, gWs);
        gT[r] = board_cycles() - a;
    }
    report("logmel_256x32", stackOf(oneMel), h[0] == HASH_MEL);

    oneMfcc();
    for (int r = 0; r < REPS; ++r)
    {
        const uint32_t a = board_cycles();
        t1::runMfcc(kMfcc, gLogMel, gCoeffs);
        gT[r] = board_cycles() - a;
    }
    report("mfcc_32x13", stackOf(oneMfcc), h[1] == HASH_MFCC);

    oneKMeans();
    for (int r = 0; r < REPS; ++r)
    {
        const uint32_t a = board_cycles();
        gSink = t1::runKMeans(kKMeans, gFeat[r % t1::kFrames]);
        gT[r] = board_cycles() - a;
    }
    report("kmeans_score_32x8", stackOf(oneKMeans), h[2] == HASH_KMEANS);

    oneGauss();
    for (int r = 0; r < REPS; ++r)
    {
        const uint32_t a = board_cycles();
        gSink = t1::runGaussian(kGauss, gFeat[r % t1::kFrames]);
        gT[r] = board_cycles() - a;
    }
    report("gaussian_score_32x4", stackOf(oneGauss), h[3] == HASH_GAUSSIAN);

    // Learning: the clusters are seeded first, so every timed call is a
    // full update (nearest centroid, then a step towards the sample).
    t1::trainerWarmup(gTrainer);
    board_stack_paint();
    for (int r = 0; r < REPS; ++r)
    {
        const uint32_t a = board_cycles();
        t1::runLearn(gTrainer, gFeat[r % t1::kFrames]);
        gT[r] = board_cycles() - a;
    }
    const uint32_t learnStack = board_stack_used();
    report("kmeans_learn_32x8", learnStack, t1::hashTrainer(gTrainer) == HASH_LEARN);
#endif

    // Stack the harness itself uses at the same point, to subtract.
    kv("stack_baseline", stackOf(none));
    kv("static_ram", board_static_ram());
    board_puts("\n# end bench\n");
    halt();
}
