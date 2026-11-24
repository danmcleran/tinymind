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

// Shared by the host generator (gen_tables.cpp) and the board image
// (tier1_bench.cpp), so both compute the outputs they compare the same way.
#pragma once

#include "tinymind_platform.hpp"
#include "qmel.hpp"
#include "qanomaly.hpp"

// The generated digits models: int8, int4 dense layer, and 16x8.
#include "model.hpp"
#include "model_int4.hpp"
#include "model_int16x8.hpp"
#include "reference.hpp"

#include <stddef.h>
#include <stdint.h>

namespace t1
{

constexpr size_t kFrames = 8;     // inputs per workload, cycled through the timed runs
constexpr size_t kMelN = 256;
constexpr size_t kBands = 32;
constexpr size_t kCoeffs = 13;
constexpr size_t kFeatures = 32;
constexpr size_t kClusters = 8;
constexpr size_t kComponents = 4;
constexpr int16_t kPreemphasis = 31785; // 0.97 in Q15

typedef tinymind::QLogMelSpectrogram<kMelN, kBands> Mel;
typedef tinymind::QMfcc<kBands, kCoeffs> Mfcc;
typedef tinymind::QKMeansAnomaly<kFeatures, kClusters> KMeans;
typedef tinymind::QKMeansAnomalyTrainer<kFeatures, kClusters> Trainer;
typedef tinymind::QDiagGaussianAnomaly<kFeatures, kComponents> Gaussian;

inline uint32_t lcg(uint32_t& state)
{
    state = state * 1664525u + 1013904223u;
    return state;
}

// FNV-1a over the bytes of v, low byte first.
template<typename T>
inline uint32_t fnv(uint32_t h, const T* v, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        uint32_t u = static_cast<uint32_t>(v[i]);
        for (size_t b = 0; b < sizeof(T); ++b)
        {
            h = (h ^ (u & 0xFFu)) * 16777619u;
            u >>= 8;
        }
    }
    return h;
}

// Frame `seed` of a vibration-like signal: two tones and noise, at a level
// that falls 6 dB per frame, so quiet frames (small block scales) are timed
// as well as loud ones.
inline void makeFrame(int16_t* x, size_t n, uint32_t seed)
{
    uint32_t state = seed * 7919u;
    const int32_t shift = static_cast<int32_t>(seed % kFrames);
    int32_t a = 0;
    int32_t b = 0;
    for (size_t i = 0; i < n; ++i)
    {
        // Integer oscillators (no float on the board): triangle waves.
        a = (a + 1311) & 0xFFFF;
        b = (b + 4096 + static_cast<int32_t>(seed) * 17) & 0xFFFF;
        const int32_t ta = (a < 32768) ? a - 16384 : 49152 - a;
        const int32_t tb = (b < 32768) ? b - 16384 : 49152 - b;
        const int32_t noise = static_cast<int32_t>(static_cast<int16_t>(lcg(state) >> 16)) >> 3;
        const int32_t v = (ta + (tb >> 1) + noise) >> shift;
        x[i] = static_cast<int16_t>((v > 32767) ? 32767 : ((v < -32768) ? -32768 : v));
    }
}

// Feature vector `seed`: around one of the clusters the host trained, with noise.
inline void makeFeatures(int8_t* x, uint32_t seed)
{
    uint32_t state = seed * 104729u;
    const int32_t cluster = static_cast<int32_t>(seed % kClusters);
    for (size_t f = 0; f < kFeatures; ++f)
    {
        const int32_t centre = ((cluster * 37 + static_cast<int32_t>(f) * 11) % 160) - 80;
        const int32_t noise = static_cast<int32_t>(lcg(state) >> 27) - 16;
        int32_t v = centre + noise;
        v = (v > 127) ? 127 : ((v < -128) ? -128 : v);
        x[f] = static_cast<int8_t>(v);
    }
}

// Each workload in its own noinline function, so the timed region is the
// workload and nothing the compiler merged into it.
__attribute__((noinline)) inline void runDigitsInt8(const int8_t* in, int8_t* out, int8_t* arena)
{
    digits::forward(in, out, arena);
}

__attribute__((noinline)) inline void runDigitsInt4(const int8_t* in, int8_t* out, int8_t* arena)
{
    digits::int4::forward(in, out, arena);
}

__attribute__((noinline)) inline void runDigits16x8(const int16_t* in, int16_t* out, int16_t* arena)
{
    digits::int16x8::forward(in, out, arena);
}

__attribute__((noinline)) inline void runMel(const Mel& mel, const int16_t* frame, int16_t* logMel,
                                             tinymind::QMelWorkspace<kMelN>& ws)
{
    mel.forward(frame, 0, logMel, ws);
}

__attribute__((noinline)) inline void runMfcc(const Mfcc& mfcc, const int16_t* logMel, int32_t* out)
{
    mfcc.forward(logMel, out);
}

__attribute__((noinline)) inline int32_t runKMeans(const KMeans& km, const int8_t* x)
{
    return km.score(x);
}

__attribute__((noinline)) inline int32_t runGaussian(const Gaussian& g, const int8_t* x)
{
    return g.score(x);
}

__attribute__((noinline)) inline void runLearn(Trainer& trainer, const int8_t* x)
{
    trainer.learn(x);
}

inline void digitsInput8(const uint8_t* pixels, int8_t* in)
{
    for (size_t p = 0; p < digits::kModelInputSize; ++p)
    {
        in[p] = digits::kPixelToInput[pixels[p]];
    }
}

inline void digitsInput16(const uint8_t* pixels, int16_t* in)
{
    for (size_t p = 0; p < digits::kModelInputSize; ++p)
    {
        in[p] = digits::kPixelToInput16[pixels[p]];
    }
}

// The trainer sequence both sides run: seed the clusters, then learn
// samples 1..33 cycling through the frames. Returns the samples learned.
inline void trainerWarmup(Trainer& trainer)
{
    trainer.reset();
    int8_t x[kFeatures];
    for (uint32_t s = 0; s < kClusters; ++s)
    {
        makeFeatures(x, 100u + s);
        trainer.learn(x);
    }
}

inline uint32_t hashTrainer(const Trainer& trainer)
{
    uint32_t h = 0x811C9DC5u;
    for (size_t k = 0; k < kClusters; ++k)
    {
        h = fnv(h, trainer.centroid(k), kFeatures);
        const uint16_t c = trainer.learnCount(k);
        h = fnv(h, &c, 1);
    }
    return h;
}

// Each workload's outputs over the kFrames inputs, hashed (FNV-1a).
inline uint32_t hashDigitsInt8(const uint8_t (*pixels)[64], int8_t* arena)
{
    uint32_t h = 0x811C9DC5u;
    for (uint32_t s = 0; s < kFrames; ++s)
    {
        int8_t in[digits::kModelInputSize];
        int8_t out[digits::kModelOutputSize];
        digitsInput8(pixels[s], in);
        runDigitsInt8(in, out, arena);
        h = fnv(h, out, digits::kModelOutputSize);
    }
    return h;
}

inline uint32_t hashDigitsInt4(const uint8_t (*pixels)[64], int8_t* arena)
{
    uint32_t h = 0x811C9DC5u;
    for (uint32_t s = 0; s < kFrames; ++s)
    {
        int8_t in[digits::kModelInputSize];
        int8_t out[digits::kModelOutputSize];
        digitsInput8(pixels[s], in);
        runDigitsInt4(in, out, arena);
        h = fnv(h, out, digits::kModelOutputSize);
    }
    return h;
}

inline uint32_t hashDigits16x8(const uint8_t (*pixels)[64], int16_t* arena)
{
    uint32_t h = 0x811C9DC5u;
    for (uint32_t s = 0; s < kFrames; ++s)
    {
        int16_t in[digits::kModelInputSize];
        int16_t out[digits::kModelOutputSize];
        digitsInput16(pixels[s], in);
        runDigits16x8(in, out, arena);
        h = fnv(h, out, digits::kModelOutputSize);
    }
    return h;
}

// mel, mfcc, kmeans and gaussian hashes, in that order.
inline void hashFeatures(const Mel& mel, const Mfcc& mfcc, const KMeans& km, const Gaussian& gauss,
                         tinymind::QMelWorkspace<kMelN>& ws, int16_t* frame, uint32_t* h)
{
    for (int i = 0; i < 4; ++i)
    {
        h[i] = 0x811C9DC5u;
    }
    for (uint32_t s = 0; s < kFrames; ++s)
    {
        int16_t logMel[kBands];
        int32_t coeffs[kCoeffs];
        makeFrame(frame, kMelN, s + 1);
        runMel(mel, frame, logMel, ws);
        h[0] = fnv(h[0], logMel, kBands);
        runMfcc(mfcc, logMel, coeffs);
        h[1] = fnv(h[1], coeffs, kCoeffs);
        int8_t x[kFeatures];
        makeFeatures(x, s + 1);
        const int32_t k = runKMeans(km, x);
        const int32_t g = runGaussian(gauss, x);
        h[2] = fnv(h[2], &k, 1);
        h[3] = fnv(h[3], &g, 1);
    }
}

} // namespace t1
