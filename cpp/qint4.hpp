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
#include "qconv2d.hpp"
#include "qdepthwiseconv2d.hpp"

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
 *   QConv2DInt4 a per-channel 2D convolution (pointwise with a 1x1 kernel)
 *               over the packed OHWI tensor; QConv2DPerChannel's results.
 *   QDepthwiseConv2DInt4
 *               QDepthwiseConv2D with the Int4Weights storage tag: the
 *               [Channels][KH][KW] tensor packed; QDepthwiseConv2D's results.
 *
 * Helpers: unpackInt4 / packInt4 (freestanding), and, under FLOAT && STD,
 * quantizeInt4PerChannel, which picks one symmetric scale per output channel
 * (max |w| / 7, so the codes are -7..7) and rounds the weights onto it.
 *
 * With int8 inputs and the Arm DSP backend (TINYMIND_ENABLE_SIMD_ARM_DSP),
 * the dot product runs on SMLAD: arm_dsp::int4DotWithZeroPoint widens eight
 * weights per word load, and with an even input count QDenseInt4 computes
 * two rows per pass (int4Dot1x2WithZeroPoint), as QConv2DInt4 does for two
 * filters over an even window. Every other build keeps the scalar loop.
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

        /**
         * The dense product shared by QDenseInt4 and QConv2DInt4's windows:
         *   output[o] = rqs[o].apply(bias[o] + sum_i (x[i] - zp) * w[o][i])
         * for NumOutputs rows of NumInputs packed int4 weights.
         */
        template<typename InputType, typename AccumType, typename OutputType,
                 std::size_t NumInputs, std::size_t NumOutputs>
        inline void denseForward(const InputType* input, const uint8_t* packed, const AccumType* biases, int32_t zp,
                                 const Requantizer<AccumType, OutputType>* rqs, OutputType* output)
        {
            std::size_t o = Int4RowPairs<InputType, AccumType, NumInputs, NumOutputs>::apply(
                input, packed, biases, zp, rqs, output);
            for (; o < NumOutputs; ++o)
            {
                AccumType acc = (biases != nullptr) ? biases[o] : static_cast<AccumType>(0);
                acc += dotInt4<InputType, AccumType>(input, packed, o * NumInputs, NumInputs, zp);
                output[o] = rqs[o].apply(acc);
            }
        }

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
            qint4_detail::denseForward<InputType, AccumulatorType, OutputType, NumInputs_, NumOutputs_>(
                input, packed, layerBiases, zp, rqs, output);
        }
    };

    /**
     * Largest weight tensor (NumFilters * KH * KW * InChannels weights, one
     * byte each once unpacked) that QConv2DInt4 unpacks onto the stack for
     * the duration of a forward() call, so that the layer runs the int8
     * convolution kernels (SMLAD on the Arm DSP backend) at their own
     * speed for the cost of one pass over its weights. A larger layer
     * computes from the packed weights directly, with no copy, more slowly.
     * Set it lower on stack-tight parts; 0 always takes the direct path.
     */
#ifndef TINYMIND_INT4_CONV_UNPACK_MAX_BYTES
#define TINYMIND_INT4_CONV_UNPACK_MAX_BYTES 1024
#endif

    /**
     * Largest kernel window (KH * KW * InChannels input elements, in bytes)
     * that QConv2DInt4's direct path gathers into a contiguous patch on the
     * stack, so that every filter runs one dot product over it (two filters
     * at a time on the Arm DSP backend). A larger window runs one dot
     * product per kernel row instead. A layer with a 1-row kernel reads its
     * windows in place at any size.
     */
#ifndef TINYMIND_INT4_CONV_PATCH_MAX_BYTES
#define TINYMIND_INT4_CONV_PATCH_MAX_BYTES 128
#endif

    namespace qint4_detail {

        template<bool Unpack>
        struct UnpackTag
        {
        };

        // Unpack n packed int4 values into int8, a byte (two values) at a time.
        inline void unpackInt4Tensor(const uint8_t* packed, std::size_t n, int8_t* values)
        {
            for (std::size_t b = 0; b < n / 2; ++b)
            {
                const uint8_t byte = packed[b];
                values[2 * b] = static_cast<int8_t>(lowNibble(byte));
                values[2 * b + 1] = static_cast<int8_t>(highNibble(byte));
            }
            if ((n & 1u) != 0u)
            {
                values[n - 1] = static_cast<int8_t>(lowNibble(packed[n / 2]));
            }
        }

    } // namespace qint4_detail

    /**
     * 2D convolution over packed int4 weights, per-channel (one Requantizer
     * per filter), NHWC, with QConv2DPerChannel's shape parameters and zero
     * point padding. It computes exactly what QConv2DPerChannel computes
     * with the same weight values stored as int8.
     *
     * weights      : int4PackedBytes(NumFilters * KH * KW * InChannels) bytes,
     *                the OHWI tensor [NumFilters][KH][KW][InChannels] packed
     *                as one tensor (so with an odd window every second
     *                filter starts mid-byte)
     * biases       : NumFilters values, or nullptr
     * requantizers : NumFilters Requantizers
     *
     * A 1x1 kernel is a pointwise convolution.
     *
     * A layer of up to TINYMIND_INT4_CONV_UNPACK_MAX_BYTES weights unpacks
     * them onto the stack and runs QConv2DPerChannel over the int8 copy
     * (UnpacksWeights). A larger one computes from the packed weights: each
     * output pixel is a QDenseInt4 over its window, gathered into a patch
     * when it fits TINYMIND_INT4_CONV_PATCH_MAX_BYTES and read in place when
     * the kernel is one row tall, or one dot product per kernel row.
     */
    template<
        typename InputStorage_,
        typename AccumType_,
        typename OutputStorage_,
        std::size_t H_,
        std::size_t W_,
        std::size_t InChannels_,
        std::size_t KH_,
        std::size_t KW_,
        std::size_t StrideH_ = 1,
        std::size_t StrideW_ = 1,
        std::size_t NumFilters_ = 1,
        std::size_t PadTop_ = 0,
        std::size_t PadBottom_ = 0,
        std::size_t PadLeft_ = 0,
        std::size_t PadRight_ = 0>
    struct QConv2DInt4
    {
        typedef InputStorage_ InputType;
        typedef AccumType_ AccumulatorType;
        typedef OutputStorage_ OutputType;

        static constexpr std::size_t OutputHeight = (H_ + PadTop_ + PadBottom_ - KH_) / StrideH_ + 1;
        static constexpr std::size_t OutputWidth  = (W_ + PadLeft_ + PadRight_ - KW_) / StrideW_ + 1;
        static constexpr std::size_t OutputSize   = OutputHeight * OutputWidth * NumFilters_;
        static constexpr std::size_t InputSize    = H_ * W_ * InChannels_;
        static constexpr std::size_t RowLength    = KW_ * InChannels_;
        static constexpr std::size_t WeightsPerFilter = KH_ * RowLength;
        static constexpr std::size_t PackedWeightBytes = int4PackedBytes(NumFilters_ * WeightsPerFilter);
        static constexpr bool UnpacksWeights = NumFilters_ * WeightsPerFilter <= TINYMIND_INT4_CONV_UNPACK_MAX_BYTES;
        // Direct path: a 1-row kernel's window is contiguous in the input; a
        // taller one is gathered when it fits TINYMIND_INT4_CONV_PATCH_MAX_BYTES.
        static constexpr bool UsePatch =
            (KH_ > 1) && (WeightsPerFilter * sizeof(InputStorage_) <= TINYMIND_INT4_CONV_PATCH_MAX_BYTES);

        typedef QConv2DPerChannel<InputStorage_, int8_t, AccumType_, OutputStorage_, H_, W_, InChannels_, KH_, KW_,
                                  StrideH_, StrideW_, NumFilters_, PadTop_, PadBottom_, PadLeft_, PadRight_>
            UnpackedLayer;

        static_assert(H_ + PadTop_ + PadBottom_ >= KH_, "Padded input height must be >= kernel height.");
        static_assert(W_ + PadLeft_ + PadRight_ >= KW_, "Padded input width must be >= kernel width.");
        static_assert(StrideH_ > 0 && StrideW_ > 0, "Strides must be > 0.");
        static_assert(InChannels_ > 0 && NumFilters_ > 0, "At least one input channel and one filter.");
        static_assert(WeightsPerFilter < (static_cast<std::size_t>(1) << 20), "Too many weights per filter for an int32 sum.");

        const uint8_t* weights;
        const AccumulatorType* biases;
        InputType input_zero_point;
        const Requantizer<AccumulatorType, OutputType>* requantizers; // [NumFilters]

        void forward(const InputType* input, OutputType* output) const
        {
            forwardImpl(input, output, qint4_detail::UnpackTag<UnpacksWeights>());
        }

    private:
        // Each layer instantiates only the path it takes.
        void forwardImpl(const InputType* input, OutputType* output, qint4_detail::UnpackTag<true>) const
        {
            int8_t unpacked[NumFilters_ * WeightsPerFilter];
            qint4_detail::unpackInt4Tensor(weights, NumFilters_ * WeightsPerFilter, unpacked);
            UnpackedLayer layer;
            layer.weights = unpacked;
            layer.biases = biases;
            layer.input_zero_point = input_zero_point;
            layer.requantizers = requantizers;
            layer.forward(input, output);
        }

        void forwardImpl(const InputType* input, OutputType* output, qint4_detail::UnpackTag<false>) const
        {
            // Locals, as in QConv2DPerChannel: an int8 store may alias *this.
            const uint8_t* const packed = weights;
            const AccumulatorType* const layerBiases = biases;
            const InputType inputZeroPoint = input_zero_point;
            const int32_t zp = static_cast<int32_t>(inputZeroPoint);
            const Requantizer<AccumulatorType, OutputType>* const rqs = requantizers;

            typename detail::ConvRowSource<InputType, H_, W_, InChannels_, KH_, StrideH_,
                                           PadTop_, PadBottom_, PadLeft_, PadRight_>::type source(input, inputZeroPoint);
            const InputType* rows[KH_];
            InputType patch[UsePatch ? WeightsPerFilter : 1];
            for (std::size_t oh = 0; oh < OutputHeight; ++oh)
            {
                source.rows(oh, rows);
                for (std::size_t ow = 0; ow < OutputWidth; ++ow)
                {
                    const std::size_t offset = ow * StrideW_ * InChannels_;
                    OutputType* const out = output + (oh * OutputWidth + ow) * NumFilters_;
                    if (KH_ == 1 || UsePatch)
                    {
                        const InputType* window = rows[0] + offset;
                        if (UsePatch)
                        {
                            for (std::size_t kh = 0; kh < KH_; ++kh)
                            {
                                for (std::size_t i = 0; i < RowLength; ++i)
                                {
                                    patch[kh * RowLength + i] = rows[kh][offset + i];
                                }
                            }
                            window = patch;
                        }
                        qint4_detail::denseForward<InputType, AccumulatorType, OutputType,
                                                   WeightsPerFilter, NumFilters_>(
                            window, packed, layerBiases, zp, rqs, out);
                        continue;
                    }
                    for (std::size_t f = 0; f < NumFilters_; ++f)
                    {
                        AccumulatorType acc = (layerBiases != nullptr) ? layerBiases[f] : static_cast<AccumulatorType>(0);
                        for (std::size_t kh = 0; kh < KH_; ++kh)
                        {
                            acc += qint4_detail::dotInt4<InputType, AccumulatorType>(
                                rows[kh] + offset, packed, f * WeightsPerFilter + kh * RowLength, RowLength, zp);
                        }
                        out[f] = rqs[f].apply(acc);
                    }
                }
            }
        }
    };

    /**
     * Depthwise 2D convolution over packed int4 weights: QDepthwiseConv2D
     * with the Int4Weights storage tag. Its weights are the [Channels][KH][KW]
     * tensor packed as one tensor (int4PackedBytes(Channels * KH * KW)
     * bytes). The layer unpacks one channel's taps before sweeping that
     * channel's pixels, so its inner loop is the int8 layer's.
     */
    template<typename InputStorage_, typename AccumType_, typename OutputStorage_,
             std::size_t H_, std::size_t W_, std::size_t Channels_, std::size_t KH_, std::size_t KW_,
             std::size_t StrideH_ = 1, std::size_t StrideW_ = 1,
             std::size_t PadTop_ = 0, std::size_t PadBottom_ = 0, std::size_t PadLeft_ = 0, std::size_t PadRight_ = 0>
    using QDepthwiseConv2DInt4 = QDepthwiseConv2D<InputStorage_, Int4Weights, AccumType_, OutputStorage_,
                                                  H_, W_, Channels_, KH_, KW_, StrideH_, StrideW_,
                                                  PadTop_, PadBottom_, PadLeft_, PadRight_>;

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
