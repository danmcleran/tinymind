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
#include "qaffine.hpp"

#include <cstddef>
#include <cstdint>

#if defined(__ARM_FEATURE_SAT)
#include <arm_acle.h>
#endif

/*
 * Quantized radix-2 decimation-in-time FFT for power-of-two lengths.
 *
 * Integer counterpart of fft1d.hpp. Designed for audio / vibration / IMU
 * front-ends running on MCU-class targets where float is unavailable or
 * too expensive. Butterfly arithmetic is int16, twiddle factors are Q1.15
 * (caller-owned), and each stage right-shifts the result by 1 to keep the
 * working register bounded -- total scaling across all stages is 1/N, the
 * same shape used by the float FFT.
 *
 * Layer carries only the twiddle pointers; the int16 work buffers are
 * caller-owned, matching the rest of the q*.hpp family. Affine boundary
 * conditions (int8 input -> int16 working domain, int32 magnitude squared
 * -> int8 output) are expressed as ordinary Requantizers built host-side
 * by the calibration helpers in qcalibration.hpp.
 *
 * Pure integer at runtime; freestanding-safe.
 */

namespace tinymind {

    namespace qfft_detail {

        template<std::size_t N>
        struct Log2
        {
            static constexpr std::size_t value = 1 + Log2<N / 2>::value;
        };

        template<>
        struct Log2<1>
        {
            static constexpr std::size_t value = 0;
        };

        /**
         * Smallest unsigned type that holds every index 0..N-1: uint8_t up
         * to N = 256, uint16_t up to N = 65536, std::size_t beyond. The
         * table lives in flash, and a std::size_t entry would cost 4 bytes
         * per index on a 32-bit MCU (1 KB at N = 256) where 1 is enough.
         */
        template<std::size_t N, bool FitsUint8 = (N <= 256u),
                 bool FitsUint16 = (N <= 65536u)>
        struct BitReversalIndex
        {
            typedef std::size_t type;
        };

        template<std::size_t N>
        struct BitReversalIndex<N, true, true>
        {
            typedef uint8_t type;
        };

        template<std::size_t N>
        struct BitReversalIndex<N, false, true>
        {
            typedef uint16_t type;
        };

        template<std::size_t N, std::size_t NumStages>
        struct BitReversalTable
        {
            typedef typename BitReversalIndex<N>::type IndexType;

            IndexType indices[N];

            constexpr BitReversalTable() : indices{}
            {
                for (std::size_t i = 0; i < N; ++i)
                {
                    std::size_t x = i;
                    std::size_t result = 0;
                    for (std::size_t b = 0; b < NumStages; ++b)
                    {
                        result = (result << 1) | (x & 1);
                        x >>= 1;
                    }
                    indices[i] = static_cast<IndexType>(result);
                }
            }
        };

        /**
         * Saturating cast int32 -> int16.
         */
        inline int16_t saturateInt16(int32_t v)
        {
#if defined(__ARM_FEATURE_SAT)
            // One SSAT. GCC 13 compiles an early-return clamp
            // (if (v < lo) return lo; ...) to two compare-and-branch pairs,
            // and does not recognize the conditional form below either.
            return static_cast<int16_t>(__ssat(v, 16));
#else
            v = (v < -32768) ? -32768 : v;
            v = (v >  32767) ?  32767 : v;
            return static_cast<int16_t>(v);
#endif
        }

        /**
         * Q1.15 fixed-point multiply with arithmetic right shift.
         *
         * Result = (a * b) >> 15. a and b are int16 values in Q1.15, so the
         * product is at most 2^30 in magnitude and the 32-bit intermediate
         * cannot overflow. The result can be +32768 (a == b == -32768;
         * buildQFFTTwiddles emits -32768 as the sine at k = N/4), which is
         * why it stays int32: the butterfly saturates its sums to int16.
         */
        inline int32_t q15Mul(int32_t a, int32_t b)
        {
            return (a * b) >> 15;
        }

        /**
         * One radix-2 DIT stage over n points with butterfly span halfSize:
         * the twiddle for pair p is entry p * twiddleStride of the tables.
         * Scale halves every output (forward); without it the sums are
         * only saturated (inverse).
         *
         * Kept out of line, and written so that only three values (the end
         * of the data and the two table bases) live across the inner loop,
         * which itself carries seven. GCC 13 otherwise keeps the stage and
         * group bookkeeping in registers and spills the butterfly's values
         * to the stack: on a Cortex-M4 that cost 78 to 114 cycles per
         * butterfly, varying by up to 45% with N and with unrelated code
         * nearby. benchmarks/fft_onboard measures it.
         */
        template<bool Scale>
        TINYMIND_NOINLINE void butterflyStage(int16_t* re, int16_t* im, std::size_t n,
                                              std::size_t halfSize,
                                              const int16_t* cosTable,
                                              const int16_t* sinTable,
                                              std::size_t twiddleStride)
        {
            const int32_t shift = Scale ? 1 : 0;
            int16_t* const reEnd = re + n;
            while (re != reEnd)
            {
                const int16_t* wc = cosTable;
                const int16_t* ws = sinTable;
                int16_t* const groupEnd = re + halfSize;

                for (; re != groupEnd; ++re, ++im, wc += twiddleStride, ws += twiddleStride)
                {
                    const int32_t wCos = static_cast<int32_t>(*wc);
                    const int32_t wSin = static_cast<int32_t>(*ws);

                    const int32_t botReal = static_cast<int32_t>(re[halfSize]);
                    const int32_t botImag = static_cast<int32_t>(im[halfSize]);

                    const int32_t tReal = q15Mul(wCos, botReal) - q15Mul(wSin, botImag);
                    const int32_t tImag = q15Mul(wCos, botImag) + q15Mul(wSin, botReal);

                    const int32_t topReal = static_cast<int32_t>(*re);
                    const int32_t topImag = static_cast<int32_t>(*im);

                    *re = saturateInt16((topReal + tReal) >> shift);
                    *im = saturateInt16((topImag + tImag) >> shift);
                    re[halfSize] = saturateInt16((topReal - tReal) >> shift);
                    im[halfSize] = saturateInt16((topImag - tImag) >> shift);
                }

                // re and im stopped at the bottom half; skip it.
                re += halfSize;
                im += halfSize;
            }
        }

        /**
         * The stages from span halfSize up to n, on bit-reversed data, with
         * twiddles from tables of tableHalf entries. Shared by QFFT1D and
         * QRFFT1D.
         */
        template<bool Scale>
        inline void butterflyStages(int16_t* real, int16_t* imag,
                                    std::size_t n, std::size_t halfSize,
                                    const int16_t* cosTable,
                                    const int16_t* sinTable,
                                    std::size_t tableHalf)
        {
            for (; halfSize < n; halfSize <<= 1)
            {
                butterflyStage<Scale>(real, imag, n, halfSize, cosTable, sinTable,
                                      tableHalf / halfSize);
            }
        }

        /**
         * One bin of QRFFT1D's split step: X[k] = E[k] + W * O[k] from
         * a = Z[k] and b = Z[M - k], with E = (a + conj(b)) / 2 and
         * O = (a - conj(b)) / (2i). A free function rather than a member,
         * so every QRFFT1D<N> shares one definition (QRFFT1D<2> has no
         * split loop and would otherwise leave its copy uncalled).
         */
        inline void qrfftSplitBin(int32_t aReal, int32_t aImag,
                                  int32_t bReal, int32_t bImag,
                                  int32_t wCos, int32_t wSin,
                                  int16_t& outReal, int16_t& outImag)
        {
            const int32_t evenReal = (aReal + bReal) >> 1;
            const int32_t evenImag = (aImag - bImag) >> 1;
            const int32_t oddReal  = (aImag + bImag) >> 1;
            const int32_t oddImag  = (bReal - aReal) >> 1;

            const int32_t tReal = q15Mul(wCos, oddReal) - q15Mul(wSin, oddImag);
            const int32_t tImag = q15Mul(wCos, oddImag) + q15Mul(wSin, oddReal);

            outReal = saturateInt16(evenReal + tReal);
            outImag = saturateInt16(evenImag + tImag);
        }

    } // namespace qfft_detail

    /**
     * Quantized radix-2 DIT FFT, in-place on int16 buffers.
     *
     * Twiddle factors are caller-owned and stored as Q1.15 int16 (scale
     * 1/32768). Half-length entries cover one cycle of -2*pi*k/N for
     * k = 0..N/2 - 1; buildQFFTTwiddles() in qcalibration.hpp emits a
     * matching table at calibration time.
     *
     * forward() uses scaled butterflies: each of NumStages stages
     * right-shifts by 1, so the total scaling is 1/N. inverse() uses the
     * conjugate trick with unscaled butterflies so that forward(inverse(.))
     * round-trips up to int16 truncation.
     */
    template<std::size_t N>
    struct QFFT1D
    {
        static constexpr std::size_t Length     = N;
        static constexpr std::size_t HalfLength = N / 2;
        static constexpr std::size_t NumStages  = qfft_detail::Log2<N>::value;

        // Q1.15 twiddle tables; HalfLength entries each.
        const int16_t* twiddle_cos;
        const int16_t* twiddle_sin;

        /**
         * Forward FFT (in-place, scaled by 1/N).
         */
        void forward(int16_t* real, int16_t* imag) const
        {
            forwardImpl(real, imag, true);
        }

        /**
         * Inverse FFT (in-place, unscaled so it cancels forward's 1/N).
         */
        void inverse(int16_t* real, int16_t* imag) const
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                imag[i] = static_cast<int16_t>(-static_cast<int32_t>(imag[i]));
            }
            forwardImpl(real, imag, false);
            for (std::size_t i = 0; i < N; ++i)
            {
                imag[i] = static_cast<int16_t>(-static_cast<int32_t>(imag[i]));
            }
        }

        /**
         * Per-bin magnitude squared. The output is in (working_scale)^2 ;
         * an output Requantizer can rescale it to the int8 output grid.
         *
         * Summed in int64: int32 head room covers each square but not the
         * sum at the INT16_MIN rails -- (-32768)^2 + (-32768)^2 == 2^31 is
         * one past INT32_MAX (signed-overflow UB found by fuzz_qfft1d).
         * The sum is non-negative and at most 2^31, so saturating to
         * INT32_MAX only alters that single both-rails bin.
         */
        static void magnitudeSquared(const int16_t* real, const int16_t* imag,
                                     int32_t* mag_sq)
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                const int64_t r = static_cast<int64_t>(real[i]);
                const int64_t m = static_cast<int64_t>(imag[i]);
                const int64_t sq = r * r + m * m;
                mag_sq[i] = (sq > static_cast<int64_t>(INT32_MAX))
                    ? INT32_MAX
                    : static_cast<int32_t>(sq);
            }
        }

    private:
        static constexpr qfft_detail::BitReversalTable<N, NumStages> sBitRevTable{};

        void forwardImpl(int16_t* real, int16_t* imag, bool scale) const
        {
            bitReversalPermute(real, imag);
            if (scale)
            {
                qfft_detail::butterflyStages<true>(real, imag, N, 1, twiddle_cos, twiddle_sin, HalfLength);
            }
            else
            {
                qfft_detail::butterflyStages<false>(real, imag, N, 1, twiddle_cos, twiddle_sin, HalfLength);
            }
        }

        static void bitReversalPermute(int16_t* real, int16_t* imag)
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                const std::size_t j = sBitRevTable.indices[i];
                if (i < j)
                {
                    int16_t tmp = real[i];
                    real[i] = real[j];
                    real[j] = tmp;
                    tmp = imag[i];
                    imag[i] = imag[j];
                    imag[j] = tmp;
                }
            }
        }

        static_assert((N & (N - 1)) == 0, "FFT length must be a power of two.");
        static_assert(N >= 2, "FFT length must be at least 2.");
    };

#if TINYMIND_CPLUSPLUS < 201703L
    // sBitRevTable is read at run time, so before C++17 (which makes a static
    // constexpr member implicitly inline) it needs this definition.
    template<std::size_t N>
    constexpr qfft_detail::BitReversalTable<N, QFFT1D<N>::NumStages> QFFT1D<N>::sBitRevTable;
#endif

    /**
     * Quantized real-input FFT: N real int16 samples in, the N/2 + 1
     * non-redundant bins out, scaled by 1/N like QFFT1D<N>::forward.
     *
     * A real signal's spectrum is conjugate-symmetric, so a complex FFT of
     * it (imaginary part zeroed) spends half its input and half its output
     * on nothing. QRFFT1D packs the N samples into N/2 complex values (even
     * samples real, odd samples imaginary), runs an N/2-point complex FFT on
     * them, and separates the two halves in a final split step:
     *
     *   E[k] = (Z[k] + conj(Z[M - k])) / 2        (spectrum of the evens)
     *   O[k] = (Z[k] - conj(Z[M - k])) / (2i)     (spectrum of the odds)
     *   X[k] = E[k] + W_N^k * O[k]                 k = 0 .. M, M = N/2
     *
     * At N = 256 that is 448 butterflies (64 of them multiply-free) and 128
     * split outputs instead of QFFT1D's 1,024 butterflies, with the same
     * twiddle table: pass the N-point table from buildQFFTTwiddles(N, ...),
     * the one QFFT1D<N> uses. The inner N/2-point FFT reads every second
     * entry of it.
     *
     * Scaling and headroom. Packing two real samples into one complex value
     * lets a component of the inner FFT reach sqrt(2) * 32767, past int16,
     * on full-scale input. So the first inner stage, whose twiddle is
     * exactly 1, computes (a +- b) >> 2 from the full-precision samples
     * instead of >> 1, and the split step does not halve its result. Every
     * component of the inner FFT then stays within 16,384 * sqrt(2), and
     * since |E[k]| and |O[k]| are each at most half of full scale in exact
     * arithmetic, the split step can reach the int16 rails only by rounding,
     * where it saturates. No input saturates an intermediate. The cost of
     * taking the headroom early is precision: against an exact DFT / N the
     * largest bin error is about 1.3 to 1.7 times QFFT1D's on the same real
     * input (a few LSB of the 16-bit spectrum), and the bins differ from
     * QFFT1D's by a few LSB. Use QFFT1D<N> with a zeroed imaginary buffer
     * when that matters more than the halved work.
     *
     * Rounding is QFFT1D's: every shift floors and every Q1.15 product goes
     * through q15Mul.
     *
     * Buffers are caller-owned: `input` holds N samples, `real` and `imag`
     * NumBins = N/2 + 1 entries each. `input` must not overlap `real` or
     * `imag`; the outputs are also the inner FFT's work space. Bin 0 (DC)
     * and bin N/2 (Nyquist) are real, so imag[0] and imag[N/2] are 0.
     *
     * Pure integer at runtime; freestanding-safe.
     */
    template<std::size_t N>
    struct QRFFT1D
    {
        static constexpr std::size_t Length     = N;
        static constexpr std::size_t HalfLength = N / 2;
        static constexpr std::size_t NumBins    = N / 2 + 1;

        // Q1.15 twiddle tables for N points (buildQFFTTwiddles(N, ...)),
        // HalfLength entries each.
        const int16_t* twiddle_cos;
        const int16_t* twiddle_sin;

        /**
         * Forward real FFT, scaled by 1/N. Writes NumBins bins to real and
         * imag.
         */
        void forward(const int16_t* input, int16_t* real, int16_t* imag) const
        {
            const std::size_t M = HalfLength;
            const int16_t* const cosTable = twiddle_cos;
            const int16_t* const sinTable = twiddle_sin;

            std::size_t halfSize = 1;
            if (M >= 2)
            {
                // Bit-reversed load of the packed samples, fused with the
                // first stage (twiddle 1): (a +- b) >> 2.
                for (std::size_t n = 0; n < M; n += 2)
                {
                    const std::size_t j0 = sBitRevTable.indices[n];
                    const std::size_t j1 = sBitRevTable.indices[n + 1];
                    const int32_t aReal = static_cast<int32_t>(input[2 * j0]);
                    const int32_t aImag = static_cast<int32_t>(input[2 * j0 + 1]);
                    const int32_t bReal = static_cast<int32_t>(input[2 * j1]);
                    const int32_t bImag = static_cast<int32_t>(input[2 * j1 + 1]);
                    real[n]     = static_cast<int16_t>((aReal + bReal) >> 2);
                    imag[n]     = static_cast<int16_t>((aImag + bImag) >> 2);
                    real[n + 1] = static_cast<int16_t>((aReal - bReal) >> 2);
                    imag[n + 1] = static_cast<int16_t>((aImag - bImag) >> 2);
                }
                halfSize = 2;
            }
            else
            {
                // N = 2: one complex value, no inner stage.
                real[0] = input[0];
                imag[0] = input[1];
            }

            // The remaining stages of the N/2-point FFT, as QFFT1D's scaled
            // butterflies; stage twiddle W_{2h}^p = W_N^{p * M / h}.
            qfft_detail::butterflyStages<true>(real, imag, M, halfSize, cosTable, sinTable, M);

            // Split step. DC and Nyquist come from Z[0] alone. For N = 2 the
            // missing first-stage halving is taken here instead.
            const int32_t z0Real = static_cast<int32_t>(real[0]);
            const int32_t z0Imag = static_cast<int32_t>(imag[0]);
            const int32_t edgeShift = (M >= 2) ? 0 : 1;
            real[0] = qfft_detail::saturateInt16((z0Real + z0Imag) >> edgeShift);
            real[M] = qfft_detail::saturateInt16((z0Real - z0Imag) >> edgeShift);
            imag[0] = 0;
            imag[M] = 0;

            for (std::size_t k = 1; 2 * k <= M; ++k)
            {
                const int32_t aReal = static_cast<int32_t>(real[k]);
                const int32_t aImag = static_cast<int32_t>(imag[k]);
                const int32_t bReal = static_cast<int32_t>(real[M - k]);
                const int32_t bImag = static_cast<int32_t>(imag[M - k]);

                qfft_detail::qrfftSplitBin(aReal, aImag, bReal, bImag,
                                           static_cast<int32_t>(cosTable[k]),
                                           static_cast<int32_t>(sinTable[k]),
                                           real[k], imag[k]);
                if (M - k != k)
                {
                    qfft_detail::qrfftSplitBin(bReal, bImag, aReal, aImag,
                                               static_cast<int32_t>(cosTable[M - k]),
                                               static_cast<int32_t>(sinTable[M - k]),
                                               real[M - k], imag[M - k]);
                }
            }
        }

        /**
         * Per-bin magnitude squared over the NumBins bins, summed in int64
         * and saturated to INT32_MAX as QFFT1D::magnitudeSquared does.
         */
        static void magnitudeSquared(const int16_t* real, const int16_t* imag,
                                     int32_t* mag_sq)
        {
            for (std::size_t i = 0; i < NumBins; ++i)
            {
                const int64_t r = static_cast<int64_t>(real[i]);
                const int64_t m = static_cast<int64_t>(imag[i]);
                const int64_t sq = r * r + m * m;
                mag_sq[i] = (sq > static_cast<int64_t>(INT32_MAX))
                    ? INT32_MAX
                    : static_cast<int32_t>(sq);
            }
        }

    private:
        static constexpr std::size_t InnerStages = qfft_detail::Log2<N / 2>::value;
        static constexpr qfft_detail::BitReversalTable<N / 2, InnerStages> sBitRevTable{};

        static_assert((N & (N - 1)) == 0, "FFT length must be a power of two.");
        static_assert(N >= 2, "FFT length must be at least 2.");
    };

#if TINYMIND_CPLUSPLUS < 201703L
    template<std::size_t N>
    constexpr qfft_detail::BitReversalTable<N / 2, QRFFT1D<N>::InnerStages> QRFFT1D<N>::sBitRevTable;
#endif

} // namespace tinymind
