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
#include "include/simd/simd_dispatch.hpp"
#include "qaffine.hpp"

#include <cstddef>
#include <cstdint>

/*
 * Packed int4 weights: two signed 4-bit weights per byte, half the flash of
 * int8 weights.
 *
 * Packing is TFLite's and ONNX's: a tensor of n int4 values is stored in
 * (n + 1) / 2 bytes, contiguously, element k in the low nibble of byte k / 2
 * when k is even and in the high nibble when k is odd. A weight matrix is
 * packed as one tensor, so with an odd number of inputs every second row
 * starts in the middle of a byte.
 *
 *   QDenseInt4  a dense layer over packed int4 weights, int8 activations and
 *               an int32 accumulator, with one Requantizer per output
 *               (int4 weights need per-channel scales; pass the same
 *               requantizer for every output for a per-tensor layer). It
 *               computes exactly what QDense / QConv2DPerChannel compute with
 *               the same weight values stored as int8.
 *
 * Helpers: unpackInt4 / packInt4 (freestanding), and, under FLOAT && STD,
 * quantizeInt4PerChannel, which picks one symmetric scale per output channel
 * (max |w| / 7, so the codes are -7..7) and rounds the weights onto it.
 *
 * With int8 inputs and the Arm DSP backend (TINYMIND_ENABLE_SIMD_ARM_DSP),
 * the dot product runs on SMLAD: arm_dsp::int4DotWithZeroPoint widens eight
 * weights per word load, and with an even input count QDenseInt4 computes
 * two rows per pass (int4Dot1x2WithZeroPoint). Every other build keeps the
 * scalar loop.
 *
 * Pure integer at runtime; freestanding-safe.
 */

namespace tinymind {

    namespace qint4_detail {

        // Sign-extend a 4-bit field without implementation-defined shifts.
        inline int32_t signExtendNibble(uint32_t nibble)
        {
            return static_cast<int32_t>(nibble ^ 8u) - 8;
        }

        inline int32_t lowNibble(uint8_t b)
        {
            return signExtendNibble(static_cast<uint32_t>(b) & 15u);
        }

        inline int32_t highNibble(uint8_t b)
        {
            return signExtendNibble(static_cast<uint32_t>(b) >> 4);
        }

        /**
         * sum_{i < n} (x[i] - zp) * w[first + i], w the int4 tensor packed in
         * `packed`, starting at element index `first` (either nibble).
         */
        template<typename InputType, typename AccumType>
        inline AccumType dotInt4(const InputType* x, const uint8_t* packed, std::size_t first,
                                 std::size_t n, int32_t zp)
        {
            int32_t acc = 0;
            const uint8_t* p = packed + first / 2;
            std::size_t i = 0;
            if ((first & 1u) != 0u && n > 0u)
            {
                acc += (static_cast<int32_t>(x[0]) - zp) * highNibble(*p);
                ++p;
                i = 1;
            }
            for (; i + 1 < n; i += 2)
            {
                const uint8_t b = *p++;
                acc += (static_cast<int32_t>(x[i]) - zp) * lowNibble(b);
                acc += (static_cast<int32_t>(x[i + 1]) - zp) * highNibble(b);
            }
            if (i < n)
            {
                acc += (static_cast<int32_t>(x[i]) - zp) * lowNibble(*p);
            }
            return static_cast<AccumType>(acc);
        }

#if defined(TINYMIND_SIMD_INT4_DOT)
        // int8 inputs on the Arm DSP backend: an odd first element (a row that
        // starts in the middle of a byte) is taken alone, the rest by the
        // SMLAD kernel from the next byte boundary. Same sum.
        template<>
        inline int32_t dotInt4<int8_t, int32_t>(const int8_t* x, const uint8_t* packed, std::size_t first,
                                                std::size_t n, int32_t zp)
        {
            int32_t acc = 0;
            const uint8_t* p = packed + first / 2;
            if ((first & 1u) != 0u && n > 0u)
            {
                acc = (static_cast<int32_t>(x[0]) - zp) * highNibble(*p);
                ++p;
                ++x;
                --n;
            }
            return acc + simd::int4DotWithZeroPoint(x, p, n, static_cast<int8_t>(zp));
        }
#endif

        /**
         * Outputs QDenseInt4 computes two at a time before its one-row loop:
         * none, except with int8 inputs on the Arm DSP backend and an even
         * input count, where every row starts at a byte boundary and a pair
         * of rows shares one widening of the input.
         */
        template<typename InputType, typename AccumType, std::size_t NumInputs, std::size_t NumOutputs>
        struct Int4RowPairs
        {
            template<typename OutputType>
            static std::size_t apply(const InputType*, const uint8_t*, const AccumType*, int32_t,
                                     const Requantizer<AccumType, OutputType>*, OutputType*)
            {
                return 0;
            }
        };

#if defined(TINYMIND_SIMD_INT4_DOT)
        template<std::size_t NumInputs, std::size_t NumOutputs>
        struct Int4RowPairs<int8_t, int32_t, NumInputs, NumOutputs>
        {
            template<typename OutputType>
            static std::size_t apply(const int8_t* input, const uint8_t* packed, const int32_t* biases, int32_t zp,
                                     const Requantizer<int32_t, OutputType>* rqs, OutputType* output)
            {
                if ((NumInputs & 1u) != 0u)
                {
                    return 0;
                }
                const std::size_t pairs = NumOutputs & ~static_cast<std::size_t>(1);
                for (std::size_t o = 0; o < pairs; o += 2)
                {
                    int32_t acc[2];
                    acc[0] = (biases != nullptr) ? biases[o] : 0;
                    acc[1] = (biases != nullptr) ? biases[o + 1] : 0;
                    const uint8_t* const w0 = packed + (o * NumInputs) / 2;
                    simd::int4Dot1x2WithZeroPoint(input, w0, w0 + NumInputs / 2, NumInputs,
                                                  static_cast<int8_t>(zp), acc);
                    output[o] = rqs[o].apply(acc[0]);
                    output[o + 1] = rqs[o + 1].apply(acc[1]);
                }
                return pairs;
            }
        };
#endif

    } // namespace qint4_detail

    /**
     * Bytes needed for n packed int4 values.
     */
    constexpr std::size_t int4PackedBytes(std::size_t n)
    {
        return (n + 1u) / 2u;
    }

    /**
     * Element `index` of a packed int4 tensor, as an int8.
     */
    inline int8_t unpackInt4(const uint8_t* packed, std::size_t index)
    {
        const uint8_t b = packed[index / 2];
        return static_cast<int8_t>(((index & 1u) != 0u) ? qint4_detail::highNibble(b) : qint4_detail::lowNibble(b));
    }

    /**
     * Pack n values in -8..7 into int4PackedBytes(n) bytes. A trailing high
     * nibble is 0. Values outside -8..7 are truncated to their low 4 bits.
     */
    inline void packInt4(const int8_t* values, std::size_t n, uint8_t* packed)
    {
        for (std::size_t b = 0; b < int4PackedBytes(n); ++b)
        {
            const uint32_t low = static_cast<uint32_t>(static_cast<uint8_t>(values[2 * b])) & 15u;
            const uint32_t high = (2 * b + 1 < n)
                ? (static_cast<uint32_t>(static_cast<uint8_t>(values[2 * b + 1])) & 15u)
                : 0u;
            packed[b] = static_cast<uint8_t>(low | (high << 4));
        }
    }

    /**
     * Dense layer over packed int4 weights:
     *
     *   out[o] = requantizers[o].apply(bias[o] + sum_i (x[i] - zp) * w[o][i])
     *
     * weights      : int4PackedBytes(NumInputs * NumOutputs) bytes, the
     *                [NumOutputs][NumInputs] matrix packed as one tensor
     * biases       : NumOutputs int32 values, or nullptr
     * requantizers : NumOutputs Requantizers
     *
     * The accumulator is int32: with 8-bit inputs and int4 weights a row
     * (|x - zp| <= 255, |w| <= 8) stays below 2^31 for any NumInputs below 2^20.
     */
    template<typename InputStorage_, typename AccumType_, typename OutputStorage_,
             std::size_t NumInputs_, std::size_t NumOutputs_>
    struct QDenseInt4
    {
        typedef InputStorage_ InputType;
        typedef AccumType_ AccumulatorType;
        typedef OutputStorage_ OutputType;

        static constexpr std::size_t InputLength = NumInputs_;
        static constexpr std::size_t OutputLength = NumOutputs_;
        static constexpr std::size_t PackedWeightBytes = int4PackedBytes(NumInputs_ * NumOutputs_);

        static_assert(NumInputs_ > 0 && NumOutputs_ > 0, "At least one input and one output.");
        static_assert(NumInputs_ < (static_cast<std::size_t>(1) << 20), "Too many inputs for an int32 row sum.");

        const uint8_t* weights;
        const AccumulatorType* biases;
        InputType input_zero_point;
        const Requantizer<AccumulatorType, OutputType>* requantizers;

        void forward(const InputType* input, OutputType* output) const
        {
            // Locals, as in QDense: an int8 store may alias *this.
            const uint8_t* const packed = weights;
            const AccumulatorType* const layerBiases = biases;
            const int32_t zp = static_cast<int32_t>(input_zero_point);
            const Requantizer<AccumulatorType, OutputType>* const rqs = requantizers;
            std::size_t o = qint4_detail::Int4RowPairs<InputType, AccumulatorType, NumInputs_, NumOutputs_>::apply(
                input, packed, layerBiases, zp, rqs, output);
            for (; o < NumOutputs_; ++o)
            {
                AccumulatorType acc = (layerBiases != nullptr) ? layerBiases[o] : static_cast<AccumulatorType>(0);
                acc += qint4_detail::dotInt4<InputType, AccumulatorType>(input, packed, o * NumInputs_, NumInputs_, zp);
                output[o] = rqs[o].apply(acc);
            }
        }
    };

} // namespace tinymind

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
#include <cmath>

namespace tinymind {

    /**
     * Symmetric per-channel int4 quantization of a [channels][per_channel]
     * float matrix: scale_c = max_i |w[c][i]| / 7 and q = round(w / scale_c)
     * in -7..7 (the symmetric range; -8 is not used). A channel of zeros gets
     * scale 1 and all-zero codes. Writes channels * per_channel int8 codes
     * (pack them with packInt4) and channels scales.
     */
    inline void quantizeInt4PerChannel(const float* weights, std::size_t channels, std::size_t per_channel,
                                       int8_t* codes, float* scales)
    {
        for (std::size_t c = 0; c < channels; ++c)
        {
            const float* w = weights + c * per_channel;
            float largest = 0.0f;
            for (std::size_t i = 0; i < per_channel; ++i)
            {
                const float a = std::fabs(w[i]);
                largest = (a > largest) ? a : largest;
            }
            const float scale = (largest > 0.0f) ? largest / 7.0f : 1.0f;
            scales[c] = scale;
            for (std::size_t i = 0; i < per_channel; ++i)
            {
                long q = std::lround(static_cast<double>(w[i]) / static_cast<double>(scale));
                q = (q < -7) ? -7 : ((q > 7) ? 7 : q);
                codes[c * per_channel + i] = static_cast<int8_t>(q);
            }
        }
    }

} // namespace tinymind
#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
