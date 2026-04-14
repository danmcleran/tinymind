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

#include <cstddef>

namespace tinymind {

    namespace detail {
        template<size_t N>
        struct Log2
        {
            static const size_t value = 1 + Log2<N / 2>::value;
        };

        template<>
        struct Log2<1>
        {
            static const size_t value = 0;
        };

        template<size_t N, size_t NumStages>
        struct BitReversalTable
        {
            size_t indices[N];

            constexpr BitReversalTable() : indices{}
            {
                for (size_t i = 0; i < N; ++i)
                {
                    size_t x = i;
                    size_t result = 0;
                    for (size_t b = 0; b < NumStages; ++b)
                    {
                        result = (result << 1) | (x & 1);
                        x >>= 1;
                    }
                    indices[i] = result;
                }
            }
        };
    }

    /**
     * Radix-2 decimation-in-time FFT for power-of-two lengths.
     *
     * Designed for embedded signal processing (vibration, audio, IMU).
     * All dimensions are compile-time template parameters with zero
     * dynamic allocation. Uses scaled butterfly stages (shift-right
     * by 1 each stage) to prevent overflow in fixed-point arithmetic.
     *
     * For Q-format types, twiddle factors (sin/cos) must be provided
     * via setTwiddleFactors() since there are no built-in trig
     * functions for fixed-point. For float/double, twiddle factors
     * can be computed at runtime with standard math functions.
     *
     * Output feeds into any TinyMind network as input:
     *   FFT1D<Q8_8, 128> fft;
     *   NeuralNetwork<Q8_8, 64, ...> mlp;  // N/2 magnitude bins
     *
     *   fft.forward(real, imag);
     *   fft.magnitudeSquared(real, imag, magSq);
     *   mlp.feedForward(magSq);
     *
     * @tparam ValueType  Numeric type (QValue or float/double)
     * @tparam N          FFT length (must be power of two, >= 2)
     */
    template<typename ValueType, size_t N>
    class FFT1D
    {
    public:
        static const size_t Length = N;
        static const size_t HalfLength = N / 2;
        static const size_t NumStages = detail::Log2<N>::value;

        FFT1D()
        {
            for (size_t i = 0; i < HalfLength; ++i)
            {
                mTwiddleCos[i] = ValueType(0);
                mTwiddleSin[i] = ValueType(0);
            }
        }

        /**
         * Set the twiddle factors (cosine and sine of -2*pi*k/N).
         *
         * For float/double, compute with std::cos/std::sin.
         * For Q-format, pre-compute externally and load here.
         *
         * @param cosTable  Array of N/2 cosine values: cos(-2*pi*k/N) for k=0..N/2-1
         * @param sinTable  Array of N/2 sine values:   sin(-2*pi*k/N) for k=0..N/2-1
         */
        void setTwiddleFactors(const ValueType* cosTable, const ValueType* sinTable)
        {
            for (size_t k = 0; k < HalfLength; ++k)
            {
                mTwiddleCos[k] = cosTable[k];
                mTwiddleSin[k] = sinTable[k];
            }
        }

        /**
         * Get a twiddle cosine factor.
         */
        ValueType getTwiddleCos(const size_t index) const { return mTwiddleCos[index]; }

        /**
         * Get a twiddle sine factor.
         */
        ValueType getTwiddleSin(const size_t index) const { return mTwiddleSin[index]; }

        /**
         * Forward FFT: time domain -> frequency domain (in-place).
         *
         * Uses scaled butterfly: each stage right-shifts results by 1
         * to prevent fixed-point overflow. Total scaling is 1/N.
         *
         * @param real  Array of N real parts (input/output)
         * @param imag  Array of N imaginary parts (input/output, init to 0 for real signals)
         */
        void forward(ValueType* real, ValueType* imag) const
        {
            bitReversalPermute(real, imag);

            size_t halfSize = 1;
            for (size_t stage = 0; stage < NumStages; ++stage)
            {
                const size_t fullSize = halfSize << 1;
                const size_t twiddleStride = HalfLength / halfSize;

                for (size_t group = 0; group < N; group += fullSize)
                {
                    for (size_t pair = 0; pair < halfSize; ++pair)
                    {
                        const size_t twiddleIdx = pair * twiddleStride;
                        const ValueType wCos = mTwiddleCos[twiddleIdx];
                        const ValueType wSin = mTwiddleSin[twiddleIdx];

                        const size_t top = group + pair;
                        const size_t bot = top + halfSize;

                        // Complex multiply: W * x[bot]
                        const ValueType tReal = wCos * real[bot] - wSin * imag[bot];
                        const ValueType tImag = wCos * imag[bot] + wSin * real[bot];

                        // Scaled butterfly: divide by 2 each stage to prevent overflow
                        const ValueType topReal = real[top];
                        const ValueType topImag = imag[top];
                        real[top] = (topReal + tReal) / 2;
                        imag[top] = (topImag + tImag) / 2;
                        real[bot] = (topReal - tReal) / 2;
                        imag[bot] = (topImag - tImag) / 2;
                    }
                }
                halfSize = fullSize;
            }
        }

        /**
         * Inverse FFT: frequency domain -> time domain (in-place).
         *
         * Conjugates input, runs forward FFT, then conjugates output.
         * The forward FFT already applies 1/N scaling through its
         * scaled butterfly stages, so no additional division is needed.
         */
        void inverse(ValueType* real, ValueType* imag) const
        {
            // Conjugate
            for (size_t i = 0; i < N; ++i)
            {
                imag[i] = ValueType(0) - imag[i];
            }

            forward(real, imag);

            // Conjugate output (forward already scaled by 1/N)
            for (size_t i = 0; i < N; ++i)
            {
                imag[i] = ValueType(0) - imag[i];
            }
        }

        /**
         * Compute magnitude squared of each frequency bin.
         * Avoids sqrt which is expensive/unavailable in fixed-point.
         *
         * @param real      Input real parts (N elements)
         * @param imag      Input imaginary parts (N elements)
         * @param magSq     Output magnitude squared (N elements)
         */
        static void magnitudeSquared(const ValueType* real, const ValueType* imag,
                                     ValueType* magSq)
        {
            for (size_t i = 0; i < N; ++i)
            {
                magSq[i] = real[i] * real[i] + imag[i] * imag[i];
            }
        }

    private:
        ValueType mTwiddleCos[HalfLength];
        ValueType mTwiddleSin[HalfLength];

        static constexpr detail::BitReversalTable<N, NumStages> sBitRevTable{};

        static void bitReversalPermute(ValueType* real, ValueType* imag)
        {
            for (size_t i = 0; i < N; ++i)
            {
                const size_t j = sBitRevTable.indices[i];
                if (i < j)
                {
                    ValueType tmp = real[i];
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
}
