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

        template<std::size_t N, std::size_t NumStages>
        struct BitReversalTable
        {
            std::size_t indices[N];

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
                    indices[i] = result;
                }
            }
        };

        /**
         * Saturating cast int32 -> int16.
         */
        inline int16_t saturateInt16(int32_t v)
        {
            if (v < -32768) return static_cast<int16_t>(-32768);
            if (v >  32767) return static_cast<int16_t>( 32767);
            return static_cast<int16_t>(v);
        }

        /**
         * Q1.15 fixed-point multiply with arithmetic right shift.
         *
         * Result = (a * b) >> 15. a and b are int16 in Q1.15. The 32-bit
         * intermediate is large enough; ::-1 corner (a == b == -32768)
         * theoretically overflows the right-shifted Q1.15 multiply but the
         * pure radix-2 twiddle table only emits values in [-32767, 32767],
         * so the corner is not reachable from buildQFFTTwiddles output.
         */
        inline int32_t q15Mul(int32_t a, int32_t b)
        {
            return (a * b) >> 15;
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
         * Per-bin magnitude squared. int32 head room covers int16 * int16
         * plus the int16 imag squared. The output is in (working_scale)^2 ;
         * an output Requantizer can rescale it to the int8 output grid.
         */
        static void magnitudeSquared(const int16_t* real, const int16_t* imag,
                                     int32_t* mag_sq)
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                const int32_t r = static_cast<int32_t>(real[i]);
                const int32_t m = static_cast<int32_t>(imag[i]);
                mag_sq[i] = r * r + m * m;
            }
        }

    private:
        static constexpr qfft_detail::BitReversalTable<N, NumStages> sBitRevTable{};

        void forwardImpl(int16_t* real, int16_t* imag, bool scale) const
        {
            bitReversalPermute(real, imag);

            std::size_t halfSize = 1;
            for (std::size_t stage = 0; stage < NumStages; ++stage)
            {
                const std::size_t fullSize = halfSize << 1;
                const std::size_t twiddleStride = HalfLength / halfSize;

                for (std::size_t group = 0; group < N; group += fullSize)
                {
                    for (std::size_t pair = 0; pair < halfSize; ++pair)
                    {
                        const std::size_t twiddleIdx = pair * twiddleStride;
                        const int32_t wCos = static_cast<int32_t>(twiddle_cos[twiddleIdx]);
                        const int32_t wSin = static_cast<int32_t>(twiddle_sin[twiddleIdx]);

                        const std::size_t top = group + pair;
                        const std::size_t bot = top + halfSize;

                        const int32_t botReal = static_cast<int32_t>(real[bot]);
                        const int32_t botImag = static_cast<int32_t>(imag[bot]);

                        const int32_t tReal =
                            qfft_detail::q15Mul(wCos, botReal) -
                            qfft_detail::q15Mul(wSin, botImag);
                        const int32_t tImag =
                            qfft_detail::q15Mul(wCos, botImag) +
                            qfft_detail::q15Mul(wSin, botReal);

                        const int32_t topReal = static_cast<int32_t>(real[top]);
                        const int32_t topImag = static_cast<int32_t>(imag[top]);

                        if (scale)
                        {
                            real[top] = qfft_detail::saturateInt16((topReal + tReal) >> 1);
                            imag[top] = qfft_detail::saturateInt16((topImag + tImag) >> 1);
                            real[bot] = qfft_detail::saturateInt16((topReal - tReal) >> 1);
                            imag[bot] = qfft_detail::saturateInt16((topImag - tImag) >> 1);
                        }
                        else
                        {
                            real[top] = qfft_detail::saturateInt16(topReal + tReal);
                            imag[top] = qfft_detail::saturateInt16(topImag + tImag);
                            real[bot] = qfft_detail::saturateInt16(topReal - tReal);
                            imag[bot] = qfft_detail::saturateInt16(topImag - tImag);
                        }
                    }
                }
                halfSize = fullSize;
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

} // namespace tinymind
