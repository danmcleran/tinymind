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

// Fixed-point log-mel spectrogram and MFCCs (cpp/qmel.hpp) of a synthetic
// machine-vibration recording, compared frame by frame with the float
// definition they implement.
//
// The recording (3,200 samples per second, 12 seconds, int16) is a motor at
// 29.5 Hz with four harmonics and broadband noise. Halfway through, a bearing
// fault starts: impacts at 107 Hz ringing a 900 Hz structural resonance. The
// overall level ramps from -50 dBFS to -6 dBFS across the run, so the front
// end sees quiet and loud frames alike.
//
// Each 256-sample frame (hop 128) goes through QLogMelSpectrogram<256, 32>
// (Hann window, 32 HTK mel bands from 5 to 1,600 Hz, no pre-emphasis) and
// QMfcc<32, 13>. The reference computes the same definition in double:
// ln( sum_k W[m][k] |DFT(x / 32768 * hann)[k]|^2 ), then the orthonormal DCT.
//
// Writes output/logmel_vibration.csv (per frame: time, fault flag, level,
// then the fixed-point and float log-mel bands), output/logmel_vibration_mfcc.csv
// and output/logmel_vibration_error.csv (|error| by how far a band sits below
// the frame's loudest band).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "qcalibration.hpp"
#include "qmel.hpp"

namespace
{
    const double kPi = 3.14159265358979323846;
    const double kSampleRate = 3200.0;
    const double kSeconds = 12.0;
    const std::size_t kN = 256;
    const std::size_t kHop = 128;
    const std::size_t kBands = 32;
    const std::size_t kCoeffs = 13;
    const std::size_t kBins = kN / 2 + 1;
    const double kFMin = 5.0;
    const double kFMax = 1600.0;
    const int kFracBits = 8;

    std::vector<int16_t> makeRecording(std::vector<double>& levelDb, std::vector<int>& fault)
    {
        const std::size_t total = static_cast<std::size_t>(kSampleRate * kSeconds);
        std::mt19937 rng(7);
        std::normal_distribution<double> gauss(0.0, 1.0);
        std::vector<int16_t> x(total);
        levelDb.resize(total);
        fault.resize(total);
        double ring = 0.0;
        double ringVelocity = 0.0;
        const double omega = 2.0 * kPi * 900.0 / kSampleRate;
        const double damping = 0.985;
        for (std::size_t n = 0; n < total; ++n)
        {
            const double t = n / kSampleRate;
            double s = 0.0;
            for (int h = 1; h <= 5; ++h)
            {
                s += std::sin(2.0 * kPi * 29.5 * h * t + 0.7 * h) / h;
            }
            s += 0.05 * gauss(rng);
            fault[n] = (t >= kSeconds / 2.0) ? 1 : 0;
            if (fault[n])
            {
                // A damped 900 Hz oscillator kicked at 107 Hz.
                const double period = kSampleRate / 107.0;
                if (std::fmod(static_cast<double>(n), period) < 1.0)
                {
                    ringVelocity += 1.0;
                }
                const double next = 2.0 * std::cos(omega) * damping * ring - damping * damping * ringVelocity;
                ringVelocity = ring;
                ring = next;
                s += 0.6 * ring;
            }
            levelDb[n] = -50.0 + 44.0 * t / kSeconds;
            const double amplitude = std::pow(10.0, levelDb[n] / 20.0) * 32767.0 / 2.6;
            x[n] = static_cast<int16_t>(std::max(-32768.0, std::min(32767.0, std::round(s * amplitude))));
        }
        return x;
    }
}

int main()
{
    std::vector<double> levelDb;
    std::vector<int> fault;
    const std::vector<int16_t> x = makeRecording(levelDb, fault);

    static int16_t window[kN];
    static int16_t twiddleCos[kN / 2];
    static int16_t twiddleSin[kN / 2];
    static uint16_t bandStart[kBands];
    static uint16_t bandLength[kBands];
    static uint16_t weights[2 * kBins];
    static int16_t dct[kCoeffs * kBands];
    tinymind::buildQHannWindow(kN, window);
    tinymind::buildQFFTTwiddles(kN, twiddleCos, twiddleSin);
    const std::size_t weightCount = tinymind::buildQMelFilterbank(kN, kSampleRate, kBands, kFMin, kFMax,
                                                                  bandStart, bandLength, weights, 2 * kBins);
    tinymind::buildQDctMatrix(kBands, kCoeffs, dct);

    // ln(E) floor of -20 (Q8: -5120); no band of this recording is that quiet.
    const tinymind::QLogMelSpectrogram<kN, kBands> mel = {window, twiddleCos, twiddleSin, bandStart, bandLength,
                                                          weights, 0, -5120};
    const tinymind::QMfcc<kBands, kCoeffs> mfcc = {dct};
    tinymind::QMelWorkspace<kN> workspace;

    std::vector<std::vector<double> > filter(kBands, std::vector<double>(kBins));
    for (std::size_t m = 0; m < kBands; ++m)
    {
        for (std::size_t k = 0; k < kBins; ++k)
        {
            filter[m][k] = tinymind::qmelFilterWeight(kN, kSampleRate, kBands, kFMin, kFMax, m, k);
        }
    }

    std::FILE* bandsCsv = std::fopen("logmel_vibration.csv", "w");
    std::FILE* mfccCsv = std::fopen("logmel_vibration_mfcc.csv", "w");
    if (bandsCsv == nullptr || mfccCsv == nullptr)
    {
        std::fprintf(stderr, "cannot write the output CSVs\n");
        return 1;
    }
    std::fprintf(bandsCsv, "frame,time_s,fault,level_dbfs");
    for (std::size_t m = 0; m < kBands; ++m)
    {
        std::fprintf(bandsCsv, ",fixed_%zu", m);
    }
    for (std::size_t m = 0; m < kBands; ++m)
    {
        std::fprintf(bandsCsv, ",float_%zu", m);
    }
    std::fprintf(bandsCsv, "\n");
    std::fprintf(mfccCsv, "frame,time_s");
    for (std::size_t c = 0; c < kCoeffs; ++c)
    {
        std::fprintf(mfccCsv, ",fixed_c%zu,float_c%zu", c, c);
    }
    std::fprintf(mfccCsv, "\n");

    // |error| buckets by level below the frame's loudest band: 0-10, ..., 40+ dB.
    std::vector<double> buckets[5];
    double worstMfccError = 0.0;
    std::size_t frames = 0;
    for (std::size_t start = 0; start + kN <= x.size(); start += kHop, ++frames)
    {
        const int16_t previous = (start > 0) ? x[start - 1] : 0;
        int16_t logMel[kBands];
        mel.forward(&x[start], previous, logMel, workspace);
        int32_t coeffs[kCoeffs];
        mfcc.forward(logMel, coeffs);

        double y[kN];
        for (std::size_t n = 0; n < kN; ++n)
        {
            y[n] = x[start + n] / 32768.0 * (0.5 - 0.5 * std::cos(2.0 * kPi * n / kN));
        }
        double power[kBins];
        for (std::size_t k = 0; k < kBins; ++k)
        {
            double re = 0.0;
            double im = 0.0;
            for (std::size_t n = 0; n < kN; ++n)
            {
                re += y[n] * std::cos(2.0 * kPi * k * n / kN);
                im -= y[n] * std::sin(2.0 * kPi * k * n / kN);
            }
            power[k] = re * re + im * im;
        }
        double reference[kBands];
        for (std::size_t m = 0; m < kBands; ++m)
        {
            double e = 0.0;
            for (std::size_t k = 0; k < kBins; ++k)
            {
                e += filter[m][k] * power[k];
            }
            reference[m] = std::log(e);
        }
        const double peak = *std::max_element(reference, reference + kBands);
        for (std::size_t m = 0; m < kBands; ++m)
        {
            const double below = (peak - reference[m]) * 10.0 / std::log(10.0);
            const std::size_t bucket = std::min<std::size_t>(4, static_cast<std::size_t>(below / 10.0));
            buckets[bucket].push_back(std::fabs(logMel[m] / 256.0 - reference[m]));
        }

        const std::size_t centre = start + kN / 2;
        std::fprintf(bandsCsv, "%zu,%.4f,%d,%.2f", frames, centre / kSampleRate, fault[centre], levelDb[centre]);
        for (std::size_t m = 0; m < kBands; ++m)
        {
            std::fprintf(bandsCsv, ",%.5f", logMel[m] / 256.0);
        }
        for (std::size_t m = 0; m < kBands; ++m)
        {
            std::fprintf(bandsCsv, ",%.5f", reference[m]);
        }
        std::fprintf(bandsCsv, "\n");

        std::fprintf(mfccCsv, "%zu,%.4f", frames, centre / kSampleRate);
        for (std::size_t c = 0; c < kCoeffs; ++c)
        {
            double r = 0.0;
            for (std::size_t m = 0; m < kBands; ++m)
            {
                r += std::sqrt((c == 0 ? 1.0 : 2.0) / kBands) * std::cos(kPi / kBands * (m + 0.5) * c) * reference[m];
            }
            const double got = coeffs[c] / static_cast<double>(1 << kFracBits);
            worstMfccError = std::max(worstMfccError, std::fabs(got - r));
            std::fprintf(mfccCsv, ",%.5f,%.5f", got, r);
        }
        std::fprintf(mfccCsv, "\n");
    }
    std::fclose(bandsCsv);
    std::fclose(mfccCsv);

    std::FILE* errorCsv = std::fopen("logmel_vibration_error.csv", "w");
    if (errorCsv == nullptr)
    {
        std::fprintf(stderr, "cannot write logmel_vibration_error.csv\n");
        return 1;
    }
    const char* labels[5] = {"0-10", "10-20", "20-30", "30-40", "40+"};
    std::fprintf(errorCsv, "db_below_peak,bands,mean_abs_error,p99_abs_error,max_abs_error\n");
    std::printf("frames: %zu of %zu samples (hop %zu), %zu filter weights, workspace %zu bytes\n",
                frames, kN, kHop, weightCount, sizeof(workspace));
    std::printf("|log-mel error| in nats, by band level below the frame's loudest band:\n");
    for (std::size_t b = 0; b < 5; ++b)
    {
        std::vector<double>& e = buckets[b];
        if (e.empty())
        {
            continue;
        }
        std::sort(e.begin(), e.end());
        double mean = 0.0;
        for (std::size_t i = 0; i < e.size(); ++i)
        {
            mean += e[i];
        }
        mean /= static_cast<double>(e.size());
        const double p99 = e[static_cast<std::size_t>(0.99 * (e.size() - 1))];
        std::fprintf(errorCsv, "%s,%zu,%.5f,%.5f,%.5f\n", labels[b], e.size(), mean, p99, e.back());
        std::printf("  %5s dB: %5zu bands  mean %.4f  p99 %.4f  max %.4f\n", labels[b], e.size(), mean, p99, e.back());
    }
    std::fclose(errorCsv);
    std::printf("worst MFCC error against the float DCT of the float log-mel: %.4f\n", worstMfccError);
    std::printf("wrote logmel_vibration.csv, logmel_vibration_mfcc.csv and logmel_vibration_error.csv\n");
    return 0;
}
