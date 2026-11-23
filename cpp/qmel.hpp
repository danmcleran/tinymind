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

#pragma once

#include "include/tinymind_platform.hpp"
#include "qfft1d.hpp"

#include <cstddef>
#include <cstdint>

/*
 * Fixed-point log-mel spectrogram and MFCC front end for audio, vibration and
 * IMU frames, built on QRFFT1D. Pure integer at runtime; freestanding-safe.
 *
 * One frame of N int16 samples becomes NumBands natural-log mel energies:
 *
 *   1. pre-emphasis   y[n] = x[n] - a * x[n - 1]          (a in Q1.15, 0 = off)
 *   2. window         z[n] = y[n] * w[n]                  (w in Q1.15)
 *   3. block scaling  u[n] = z[n] * 2^-r, r chosen per frame so the largest
 *                     |u[n]| lies in [2^14, 2^15); steps 1 and 2 are exact
 *                     in 64 bits, so this is the only rounding before the FFT
 *   4. real FFT       X = QRFFT1D<N>(u), scaled by 1/N
 *   5. mel energies   E[m] = sum_k W[m][k] * |X[k]|^2     (W in Q0.16, 64-bit sum)
 *   6. log            log_mel[m] = ln(E[m]) with the scaling of steps 2 to 5
 *                     removed, in Q(LogFracBits)
 *
 * The block scaling is what keeps a quiet frame from vanishing into the
 * int16 FFT's rounding: every frame enters the FFT at full scale, and the
 * 2^-r it was scaled by comes back out as an exact integer offset of the
 * base-2 logarithm. The result is the float definition
 *
 *   log_mel[m] = ln( sum_k W[m][k] * |DFT(y * w / 32768)[k]|^2 )
 *
 * that is, the natural log of the mel-weighted power spectrum of the
 * windowed frame, with samples read as int16 / 32768 (full scale 1.0). This
 * is torchaudio's MelSpectrogram (power 2, HTK mel scale, no filter
 * normalization, no centering) followed by a natural log, and
 * buildQMelFilterbank builds torchaudio's filters.
 *
 * A band whose energy is 0 (a silent frame, or a band too narrow to cover
 * any FFT bin) and any band below log_floor reads log_floor, which plays the
 * role of the epsilon in log(E + eps).
 *
 * QMfcc applies an orthonormal DCT-II (torchaudio's create_dct with
 * norm "ortho") to the log-mel bands, giving MFCCs in the same Q format, in
 * int32 because coefficient 0 is sqrt(NumBands) times the mean log energy.
 *
 * Every table is built on the host (buildQHannWindow, buildQFFTTwiddles,
 * buildQMelFilterbank, buildQDctMatrix) and lives in flash; the layer
 * objects hold only pointers to them. A frame needs a caller-owned
 * QMelWorkspace<N>: N + 2 * (N / 2 + 1) int16 values, about 4N bytes.
 */

namespace tinymind {

    namespace qmel_detail {

        inline int32_t countLeadingZeros32(uint32_t x)
        {
#if defined(__GNUC__) || defined(__clang__)
            return (x == 0u) ? 32 : static_cast<int32_t>(__builtin_clz(x));
#else
            int32_t n = 0;
            for (uint32_t bit = static_cast<uint32_t>(1) << 31; bit != 0u && (x & bit) == 0u; bit >>= 1)
            {
                ++n;
            }
            return n;
#endif
        }

        inline int32_t countLeadingZeros64(uint64_t x)
        {
            const uint32_t high = static_cast<uint32_t>(x >> 32);
            if (high != 0u)
            {
                return countLeadingZeros32(high);
            }
            return 32 + countLeadingZeros32(static_cast<uint32_t>(x));
        }

        /**
         * log2(1 + i / 32) in Q2.30 for i = 0..32. Linear interpolation
         * between entries is within 1.8e-4 of log2 everywhere, a twentieth
         * of the Q8 output's step once converted to a natural log. A
         * template so that the C++14 out-of-class definition below may sit in
         * a header included by several translation units.
         */
        template<int Unused = 0>
        struct Log2Table
        {
            static constexpr uint32_t values[33] = {
                0u, 47667823u, 93912511u, 138816582u,
                182455581u, 224898839u, 266210141u, 306448299u,
                345667660u, 383918542u, 421247625u, 457698295u,
                493310944u, 528123241u, 562170370u, 595485245u,
                628098702u, 660039669u, 691335320u, 722011213u,
                752091421u, 781598637u, 810554283u, 838978604u,
                866890747u, 894308843u, 921250079u, 947730758u,
                973766362u, 999371606u, 1024560487u, 1049346328u,
                1073741824u
            };
        };

        /**
         * log2(x) in Q.30 for x > 0: the integer part from the position of
         * the highest set bit, the fraction from Log2Table with linear
         * interpolation on the 32 bits below the 5 that select the entry.
         */
        inline int64_t log2Q30(uint64_t x)
        {
            const int32_t msb = 63 - countLeadingZeros64(x);
            const uint64_t normalized = x << (63 - msb);
            const uint32_t index = static_cast<uint32_t>(normalized >> 58) & 31u;
            const uint64_t remainder = (normalized >> 26) & 0xFFFFFFFFull;
            const uint32_t low = Log2Table<>::values[index];
            const uint32_t high = Log2Table<>::values[index + 1];
            const uint32_t fraction = low + static_cast<uint32_t>((static_cast<uint64_t>(high - low) * remainder) >> 32);
            return (static_cast<int64_t>(msb) << 30) + static_cast<int64_t>(fraction);
        }

        // ln(2) in Q1.31.
        static constexpr int64_t kLn2Q31 = 1488522236;

        /**
         * ln(2^(log2_q30 / 2^30)) in Q(FracBits), rounded to nearest.
         */
        template<int FracBits>
        inline int32_t naturalLogFromLog2Q30(int64_t log2_q30)
        {
            const int64_t log2_q16 = log2_q30 >> 14;
            const int shift = 16 + 31 - FracBits;
            return static_cast<int32_t>((log2_q16 * kLn2Q31 + (static_cast<int64_t>(1) << (shift - 1))) >> shift);
        }

    } // namespace qmel_detail

#if TINYMIND_CPLUSPLUS < 201703L
    // Log2Table::values is indexed at run time, so before C++17 (which makes a
    // static constexpr member implicitly inline) it needs this definition.
    template<int Unused>
    constexpr uint32_t qmel_detail::Log2Table<Unused>::values[33];
#endif

    /**
     * Caller-owned scratch for one QLogMelSpectrogram<N, ...>::forward call:
     * the block-scaled frame and the FFT's bins. Its contents need not be kept
     * between frames.
     */
    template<std::size_t N>
    struct QMelWorkspace
    {
        int16_t frame[N];
        int16_t real[N / 2 + 1];
        int16_t imag[N / 2 + 1];
    };

    /**
     * Log-mel spectrogram of one frame (see the file comment).
     *
     * N           : frame and FFT length, a power of two (at least 4)
     * NumBands    : mel bands
     * LogFracBits : fractional bits of the int16 output (default 8: steps of
     *               1/256 nat, range -128 to +128)
     *
     * Fields (all tables host-built, typically constexpr arrays in flash):
     *   window       N Q1.15 values (buildQHannWindow)
     *   twiddle_cos / twiddle_sin  N / 2 Q1.15 values each (buildQFFTTwiddles(N))
     *   band_start / band_length   NumBands entries: the first FFT bin of
     *                each band and how many bins it covers
     *   weights      the bands' Q0.16 weights, band after band, sum of
     *                band_length entries (buildQMelFilterbank)
     *   preemphasis  Q1.15 coefficient, 0 for none (0.97 is 31785)
     *   log_floor    the smallest output, in Q(LogFracBits)
     */
    template<std::size_t N, std::size_t NumBands, int LogFracBits = 8>
    struct QLogMelSpectrogram
    {
        static constexpr std::size_t FrameLength = N;
        static constexpr std::size_t NumBins = N / 2 + 1;
        static constexpr std::size_t Bands = NumBands;
        static constexpr int FracBits = LogFracBits;

        static_assert((N & (N - 1)) == 0 && N >= 4, "Frame length must be a power of two, at least 4.");
        static_assert(N / 2 + 1 <= 65535u, "Bins must fit the uint16 band tables.");
        static_assert(NumBands > 0, "At least one mel band.");
        static_assert(LogFracBits >= 0 && LogFracBits <= 14, "LogFracBits must be in 0..14.");

        const int16_t* window;
        const int16_t* twiddle_cos;
        const int16_t* twiddle_sin;
        const uint16_t* band_start;
        const uint16_t* band_length;
        const uint16_t* weights;
        int16_t preemphasis;
        int16_t log_floor;

        /**
         * frame: N int16 samples. previous_sample: the sample before
         * frame[0] (the last sample of the previous hop's input, or 0), used
         * only by pre-emphasis. log_mel: NumBands outputs.
         */
        void forward(const int16_t* frame, int16_t previous_sample, int16_t* log_mel,
                     QMelWorkspace<N>& workspace) const
        {
            const int64_t a = static_cast<int64_t>(preemphasis);
            const int16_t* const w = window;

            // Pass 1: the largest windowed sample, which sets the block
            // scale. Both steps are exact: y is kept in Q15
            // (x * 2^15 - a * x_prev, |y| <= 2^31) and z = y * w in 64 bits
            // (|z| < 2^46), so a quiet, pre-emphasized frame keeps the
            // fractions of an LSB that rounding y or z would lose, and a
            // sample the window removes (a pre-emphasis step at n = 0) does
            // not set the scale for the rest of the frame.
            int64_t prev = static_cast<int64_t>(previous_sample);
            uint64_t maxAbs = 0;
            for (std::size_t n = 0; n < N; ++n)
            {
                const int64_t x = static_cast<int64_t>(frame[n]);
                const int64_t z = (x * 32768 - a * prev) * static_cast<int64_t>(w[n]);
                prev = x;
                const uint64_t mag = static_cast<uint64_t>(z < 0 ? -z : z);
                maxAbs = (mag > maxAbs) ? mag : maxAbs;
            }

            const int32_t floorValue = static_cast<int32_t>(log_floor);
            if (maxAbs == 0u)
            {
                for (std::size_t m = 0; m < NumBands; ++m)
                {
                    log_mel[m] = static_cast<int16_t>(floorValue);
                }
                return;
            }

            // Pass 2: recompute z[n] and shift it right by r so that the
            // largest |u| = |z * 2^-r| is in [2^14, 2^15).
            const int32_t shift = (63 - qmel_detail::countLeadingZeros64(maxAbs)) - 14;
            prev = static_cast<int64_t>(previous_sample);
            for (std::size_t n = 0; n < N; ++n)
            {
                const int64_t x = static_cast<int64_t>(frame[n]);
                const int64_t z = (x * 32768 - a * prev) * static_cast<int64_t>(w[n]);
                prev = x;
                workspace.frame[n] = static_cast<int16_t>((shift >= 0) ? (z >> shift)
                                                                       : (z * (static_cast<int64_t>(1) << -shift)));
            }

            QRFFT1D<N> fft;
            fft.twiddle_cos = twiddle_cos;
            fft.twiddle_sin = twiddle_sin;
            fft.forward(workspace.frame, workspace.real, workspace.imag);

            // In float terms u = (y * w / 32768) * 2^(45 - r), so
            // E_int = sum W * |X|^2 = E * 2^(16 + 90 - 2r) / N^2 and
            // log2(E) = log2(E_int) - 106 + 2r + 2 log2(N).
            // A multiply, not << 30: the offset is negative, and shifting a
            // negative value left is undefined before C++20.
            const int64_t offsetQ30 = static_cast<int64_t>(
                2 * static_cast<int32_t>(qfft_detail::Log2<N>::value) - 106 + 2 * shift) * (static_cast<int64_t>(1) << 30);
            const uint16_t* bandWeight = weights;
            for (std::size_t m = 0; m < NumBands; ++m)
            {
                const std::size_t start = band_start[m];
                const std::size_t length = band_length[m];
                uint64_t energy = 0;
                for (std::size_t j = 0; j < length; ++j)
                {
                    const int32_t re = static_cast<int32_t>(workspace.real[start + j]);
                    const int32_t im = static_cast<int32_t>(workspace.imag[start + j]);
                    const uint32_t power = static_cast<uint32_t>(re * re) + static_cast<uint32_t>(im * im);
                    energy += static_cast<uint64_t>(bandWeight[j]) * power;
                }
                bandWeight += length;

                int32_t out = floorValue;
                if (energy != 0u)
                {
                    const int32_t value = qmel_detail::naturalLogFromLog2Q30<LogFracBits>(
                        qmel_detail::log2Q30(energy) + offsetQ30);
                    out = (value < floorValue) ? floorValue : value;
                    out = (out > 32767) ? 32767 : out;
                }
                log_mel[m] = static_cast<int16_t>(out);
            }
        }
    };

    /**
     * MFCCs from log-mel bands: an orthonormal DCT-II, coefficient c of
     * NumCoeffs, as int32 in the log-mel input's Q format.
     *
     * dct: NumCoeffs x NumBands Q1.15 values, row-major (buildQDctMatrix).
     */
    template<std::size_t NumBands, std::size_t NumCoeffs>
    struct QMfcc
    {
        static_assert(NumCoeffs > 0 && NumCoeffs <= NumBands, "1 to NumBands coefficients.");

        const int16_t* dct;

        void forward(const int16_t* log_mel, int32_t* mfcc) const
        {
            const int16_t* row = dct;
            for (std::size_t c = 0; c < NumCoeffs; ++c)
            {
                int64_t acc = 0;
                for (std::size_t m = 0; m < NumBands; ++m)
                {
                    acc += static_cast<int64_t>(row[m]) * static_cast<int64_t>(log_mel[m]);
                }
                mfcc[c] = static_cast<int32_t>((acc + (static_cast<int64_t>(1) << 14)) >> 15);
                row += NumBands;
            }
        }
    };

} // namespace tinymind

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
#include <cmath>

namespace tinymind {

    /**
     * Hann window in Q1.15, 1.0 stored as 32767. periodic (the default, as
     * torch.hann_window) divides by n; symmetric by n - 1.
     */
    inline void buildQHannWindow(std::size_t n, int16_t* out, bool periodic = true)
    {
        const double pi = 3.14159265358979323846;
        const double denominator = static_cast<double>(periodic ? n : n - 1);
        for (std::size_t i = 0; i < n; ++i)
        {
            const double w = 0.5 - 0.5 * std::cos(2.0 * pi * static_cast<double>(i) / denominator);
            long q = std::lround(w * 32768.0);
            q = (q > 32767) ? 32767 : q;
            out[i] = static_cast<int16_t>(q);
        }
    }

    inline double qmelHzToMel(double hz)
    {
        return 2595.0 * std::log10(1.0 + hz / 700.0);
    }

    inline double qmelMelToHz(double mel)
    {
        return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
    }

    /**
     * The float weight of FFT bin k in mel band m: torchaudio's triangular
     * filters on the HTK mel scale, with no normalization (melscale_fbanks
     * with norm None and mel_scale "htk"). Band m rises from mel point m to
     * m + 1 and falls to m + 2, the num_bands + 2 points evenly spaced in
     * mel from f_min to f_max.
     */
    inline double qmelFilterWeight(std::size_t n_fft, double sample_rate, std::size_t num_bands,
                                   double f_min, double f_max, std::size_t band, std::size_t bin)
    {
        const double melMin = qmelHzToMel(f_min);
        const double melMax = qmelHzToMel(f_max);
        const double step = (melMax - melMin) / static_cast<double>(num_bands + 1);
        const double lower = qmelMelToHz(melMin + step * static_cast<double>(band));
        const double center = qmelMelToHz(melMin + step * static_cast<double>(band + 1));
        const double upper = qmelMelToHz(melMin + step * static_cast<double>(band + 2));
        const double f = static_cast<double>(bin) * sample_rate / static_cast<double>(n_fft);
        const double rising = (f - lower) / (center - lower);
        const double falling = (upper - f) / (upper - center);
        const double w = (rising < falling) ? rising : falling;
        return (w > 0.0) ? w : 0.0;
    }

    /**
     * Build QLogMelSpectrogram's filter tables: band_start and band_length
     * (num_bands entries each) and the Q0.16 weights, band after band, for
     * an n_fft-point FFT at sample_rate. Each band covers the bins whose
     * rounded weight is nonzero. Returns the number of weights written, or
     * 0 if they would exceed weights_capacity (2 * (n_fft / 2 + 1) always
     * suffices: each bin lies in at most two bands). A band too narrow to
     * reach any bin gets band_length 0 and always reads log_floor.
     */
    inline std::size_t buildQMelFilterbank(std::size_t n_fft, double sample_rate, std::size_t num_bands,
                                           double f_min, double f_max,
                                           uint16_t* band_start, uint16_t* band_length,
                                           uint16_t* weights, std::size_t weights_capacity)
    {
        const std::size_t numBins = n_fft / 2 + 1;
        std::size_t used = 0;
        for (std::size_t m = 0; m < num_bands; ++m)
        {
            std::size_t first = numBins;
            std::size_t last = 0;
            for (std::size_t k = 0; k < numBins; ++k)
            {
                const long q = std::lround(qmelFilterWeight(n_fft, sample_rate, num_bands, f_min, f_max, m, k) * 65536.0);
                if (q > 0)
                {
                    first = (k < first) ? k : first;
                    last = k;
                }
            }
            if (first == numBins)
            {
                band_start[m] = 0;
                band_length[m] = 0;
                continue;
            }
            const std::size_t length = last - first + 1;
            if (used + length > weights_capacity)
            {
                return 0;
            }
            band_start[m] = static_cast<uint16_t>(first);
            band_length[m] = static_cast<uint16_t>(length);
            for (std::size_t k = first; k <= last; ++k)
            {
                long q = std::lround(qmelFilterWeight(n_fft, sample_rate, num_bands, f_min, f_max, m, k) * 65536.0);
                q = (q > 65535) ? 65535 : q;
                weights[used++] = static_cast<uint16_t>(q);
            }
        }
        return used;
    }

    /**
     * Orthonormal DCT-II matrix in Q1.15, num_coeffs rows of num_bands:
     * row c is cos(pi / M * (m + 1/2) * c) times sqrt(1 / M) for c = 0 and
     * sqrt(2 / M) otherwise (torchaudio's create_dct with norm "ortho").
     */
    inline void buildQDctMatrix(std::size_t num_bands, std::size_t num_coeffs, int16_t* out)
    {
        const double pi = 3.14159265358979323846;
        const double m = static_cast<double>(num_bands);
        for (std::size_t c = 0; c < num_coeffs; ++c)
        {
            const double scale = std::sqrt(((c == 0) ? 1.0 : 2.0) / m);
            for (std::size_t b = 0; b < num_bands; ++b)
            {
                const double v = scale * std::cos(pi / m * (static_cast<double>(b) + 0.5) * static_cast<double>(c));
                long q = std::lround(v * 32768.0);
                q = (q > 32767) ? 32767 : q;
                q = (q < -32768) ? -32768 : q;
                out[c * num_bands + b] = static_cast<int16_t>(q);
            }
        }
    }

} // namespace tinymind
#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
