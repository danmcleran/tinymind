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

// Host side: builds every table the board image needs (the log-mel window,
// twiddles, filterbank and DCT; a k-means model trained here; a diagonal
// Gaussian), reads the first test digits, and writes the output hashes the
// board must reproduce.
// usage: gen_tables <optdigits.tes> <out.h>
#include "tinymind_platform.hpp"
#include "qcalibration.hpp"

#include "tier1_common.hpp"

#include <cstdio>
#include <vector>

namespace
{

template<typename T>
void array(FILE* f, const char* type, const char* name, const T* v, size_t n)
{
    std::fprintf(f, "static const %s %s[%zu] = {", type, name, n);
    for (size_t i = 0; i < n; ++i)
    {
        std::fprintf(f, "%s%lld", i ? "," : "", static_cast<long long>(v[i]));
    }
    std::fprintf(f, "};\n");
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::fprintf(stderr, "usage: gen_tables <optdigits.tes> <out.h>\n");
        return 1;
    }

    // The first kFrames test digits.
    uint8_t pixels[t1::kFrames][64];
    FILE* in = std::fopen(argv[1], "r");
    if (in == nullptr)
    {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }
    for (size_t n = 0; n < t1::kFrames; ++n)
    {
        for (size_t p = 0; p < 64; ++p)
        {
            int v = 0;
            if (std::fscanf(in, "%d,", &v) != 1 || v < 0 || v > 16)
            {
                std::fprintf(stderr, "bad pixel in row %zu\n", n);
                return 1;
            }
            pixels[n][p] = static_cast<uint8_t>(v);
        }
        int label = 0;
        if (std::fscanf(in, "%d", &label) != 1)
        {
            std::fprintf(stderr, "missing label in row %zu\n", n);
            return 1;
        }
    }
    std::fclose(in);

    // Log-mel front end: 256 samples at 3.2 kHz, 32 bands from 5 to 1600 Hz
    // (examples/vibration_anomaly), and 13 MFCCs.
    std::vector<int16_t> window(t1::kMelN);
    std::vector<int16_t> twCos(t1::kMelN / 2);
    std::vector<int16_t> twSin(t1::kMelN / 2);
    std::vector<uint16_t> bandStart(t1::kBands);
    std::vector<uint16_t> bandLength(t1::kBands);
    std::vector<uint16_t> weights(2 * (t1::kMelN / 2 + 1));
    std::vector<int16_t> dct(t1::kBands * t1::kCoeffs);
    tinymind::buildQHannWindow(t1::kMelN, window.data());
    tinymind::buildQFFTTwiddles(t1::kMelN, twCos.data(), twSin.data());
    const size_t used = tinymind::buildQMelFilterbank(t1::kMelN, 3200.0, t1::kBands, 5.0, 1600.0, bandStart.data(),
                                                      bandLength.data(), weights.data(), weights.size());
    weights.resize(used);
    tinymind::buildQDctMatrix(t1::kBands, t1::kCoeffs, dct.data());
    const t1::Mel mel = {window.data(), twCos.data(), twSin.data(), bandStart.data(), bandLength.data(),
                         weights.data(), t1::kPreemphasis, -32768};
    const t1::Mfcc mfcc = {dct.data()};

    // k-means, trained and calibrated here as a device would.
    static t1::Trainer trainer;
    int8_t x[t1::kFeatures];
    for (uint32_t s = 0; s < 400; ++s)
    {
        t1::makeFeatures(x, 1000u + s);
        trainer.learn(x);
    }
    for (uint32_t s = 0; s < 200; ++s)
    {
        t1::makeFeatures(x, 5000u + s);
        trainer.calibrate(x);
    }
    const t1::KMeans trained = trainer.model();
    std::vector<int8_t> centroids(trained.centroids, trained.centroids + t1::kClusters * t1::kFeatures);
    std::vector<uint32_t> invRadius(trained.inv_radius_sq, trained.inv_radius_sq + t1::kClusters);
    std::vector<uint16_t> featureWeight(trained.feature_weight, trained.feature_weight + t1::kFeatures);
    const t1::KMeans km = {centroids.data(), invRadius.data(), featureWeight.data()};

    // A diagonal Gaussian mixture over the first kComponents clusters.
    std::vector<double> means(t1::kComponents * t1::kFeatures);
    std::vector<double> variances(t1::kComponents * t1::kFeatures, 85.0);
    std::vector<double> mixWeights(t1::kComponents, 1.0 / t1::kComponents);
    for (size_t k = 0; k < t1::kComponents; ++k)
    {
        for (size_t f = 0; f < t1::kFeatures; ++f)
        {
            means[k * t1::kFeatures + f] = static_cast<double>(((k * 37 + f * 11) % 160)) - 80.0;
        }
    }
    std::vector<int8_t> gMeans(t1::kComponents * t1::kFeatures);
    std::vector<uint32_t> gPrecision(t1::kComponents * t1::kFeatures);
    std::vector<int32_t> gBias(t1::kComponents);
    tinymind::buildQDiagGaussianAnomaly(t1::kComponents, t1::kFeatures, means.data(), variances.data(),
                                        mixWeights.data(), 1.0f, 0, 0.25, gMeans.data(), gPrecision.data(),
                                        gBias.data());
    const t1::Gaussian gauss = {gMeans.data(), gPrecision.data(), gBias.data()};

    static int8_t arena8[digits::kModelArenaSize];
    static int16_t arena16[digits::int16x8::kModelArenaSize / sizeof(int16_t)];
    static tinymind::QMelWorkspace<t1::kMelN> ws;
    static int16_t frame[t1::kMelN];
    const uint32_t hInt8 = t1::hashDigitsInt8(pixels, arena8);
    const uint32_t hInt4 = t1::hashDigitsInt4(pixels, arena8);
    const uint32_t h16x8 = t1::hashDigits16x8(pixels, arena16);
    uint32_t hf[4];
    t1::hashFeatures(mel, mfcc, km, gauss, ws, frame, hf);

    // The board's learning sequence: warm up, then REPS learn() calls.
    t1::trainerWarmup(trainer);
    for (uint32_t r = 0; r < 33; ++r)
    {
        t1::makeFeatures(x, 1u + r % t1::kFrames);
        t1::runLearn(trainer, x);
    }
    const uint32_t hLearn = t1::hashTrainer(trainer);

    FILE* f = std::fopen(argv[2], "w");
    if (f == nullptr)
    {
        std::fprintf(stderr, "cannot write %s\n", argv[2]);
        return 1;
    }
    std::fprintf(f, "// Generated by gen_tables.cpp; do not edit.\n#pragma once\n#include <stdint.h>\n");
    std::fprintf(f, "static const uint8_t kDigitPixels[%zu][64] = {", t1::kFrames);
    for (size_t n = 0; n < t1::kFrames; ++n)
    {
        std::fprintf(f, "%s{", n ? "," : "");
        for (size_t p = 0; p < 64; ++p)
        {
            std::fprintf(f, "%s%d", p ? "," : "", pixels[n][p]);
        }
        std::fprintf(f, "}");
    }
    std::fprintf(f, "};\n");
    array(f, "int16_t", "kWindow", window.data(), window.size());
    array(f, "int16_t", "kTwCos", twCos.data(), twCos.size());
    array(f, "int16_t", "kTwSin", twSin.data(), twSin.size());
    array(f, "uint16_t", "kBandStart", bandStart.data(), bandStart.size());
    array(f, "uint16_t", "kBandLength", bandLength.data(), bandLength.size());
    array(f, "uint16_t", "kMelWeights", weights.data(), weights.size());
    array(f, "int16_t", "kDct", dct.data(), dct.size());
    array(f, "int8_t", "kCentroids", centroids.data(), centroids.size());
    array(f, "uint32_t", "kInvRadius", invRadius.data(), invRadius.size());
    array(f, "uint16_t", "kFeatureWeight", featureWeight.data(), featureWeight.size());
    array(f, "int8_t", "kGaussMeans", gMeans.data(), gMeans.size());
    array(f, "uint32_t", "kGaussPrecision", gPrecision.data(), gPrecision.size());
    array(f, "int32_t", "kGaussBias", gBias.data(), gBias.size());
    std::fprintf(f, "#define HASH_INT8 0x%08Xu\n#define HASH_INT4 0x%08Xu\n#define HASH_16X8 0x%08Xu\n"
                    "#define HASH_MEL 0x%08Xu\n#define HASH_MFCC 0x%08Xu\n#define HASH_KMEANS 0x%08Xu\n"
                    "#define HASH_GAUSSIAN 0x%08Xu\n#define HASH_LEARN 0x%08Xu\n",
                 hInt8, hInt4, h16x8, hf[0], hf[1], hf[2], hf[3], hLearn);
    std::fclose(f);
    std::printf("int8 %08X int4 %08X 16x8 %08X mel %08X mfcc %08X kmeans %08X gaussian %08X learn %08X\n",
                hInt8, hInt4, h16x8, hf[0], hf[1], hf[2], hf[3], hLearn);
    return 0;
}
