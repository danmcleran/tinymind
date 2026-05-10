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

#include <cstddef>
#include <cstdint>

/*
 * Quantized 2D pooling layers (NHWC).
 *
 * Affine quantization is exactly linear, so as long as the input and
 * output use the same (scale, zero_point) the integer pooling math
 * matches the float reference up to rounding:
 *
 *   real = scale * (q - zp)
 *   max(real)  -> max(q)            (monotonic; trivial)
 *   mean(real) -> round(mean(q))    (zp constant cancels around the mean)
 *
 * QMaxPool2D therefore needs no requantizer at all; QAvgPool2D and
 * QGlobalAvgPool2D round the integer sum to the nearest grid point and
 * clamp to the destination storage range. Callers that need a different
 * output scale should follow with a Requantizer step.
 */

namespace tinymind {

    namespace detail {

        /**
         * Symmetric round-half-away-from-zero integer divide. Implemented
         * locally so the pooling layers stay header-only and freestanding.
         */
        inline int32_t roundedDivide(int32_t numerator, int32_t denominator)
        {
            if (denominator == 0)
            {
                return 0;
            }
            const int32_t half = denominator / 2;
            return (numerator >= 0)
                ? (numerator + half) / denominator
                : (numerator - half) / denominator;
        }

    } // namespace detail

    /**
     * Max pooling. Storage is typically int8_t; the comparison preserves
     * the affine ordering so input and output share the same (scale, zp).
     */
    template<
        typename Storage_,
        std::size_t H_,
        std::size_t W_,
        std::size_t Channels_,
        std::size_t PoolH_,
        std::size_t PoolW_,
        std::size_t StrideH_ = PoolH_,
        std::size_t StrideW_ = PoolW_>
    struct QMaxPool2D
    {
        typedef Storage_ StorageType;

        static constexpr std::size_t OutputHeight = (H_ - PoolH_) / StrideH_ + 1;
        static constexpr std::size_t OutputWidth  = (W_ - PoolW_) / StrideW_ + 1;
        static constexpr std::size_t OutputSize   = OutputHeight * OutputWidth * Channels_;
        static constexpr std::size_t InputSize    = H_ * W_ * Channels_;

        static_assert(H_ >= PoolH_, "Input height must be >= pool height.");
        static_assert(W_ >= PoolW_, "Input width must be >= pool width.");
        static_assert(StrideH_ > 0 && StrideW_ > 0, "Strides must be > 0.");
        static_assert(PoolH_ > 0 && PoolW_ > 0, "Pool dims must be > 0.");
        static_assert(Channels_ > 0, "Channels must be > 0.");

        void forward(const StorageType* input, StorageType* output) const
        {
            for (std::size_t oh = 0; oh < OutputHeight; ++oh)
            {
                const std::size_t ihStart = oh * StrideH_;
                for (std::size_t ow = 0; ow < OutputWidth; ++ow)
                {
                    const std::size_t iwStart = ow * StrideW_;
                    const std::size_t outPixelOffset =
                        (oh * OutputWidth + ow) * Channels_;

                    for (std::size_t c = 0; c < Channels_; ++c)
                    {
                        const std::size_t firstInIdx =
                            (ihStart * W_ + iwStart) * Channels_ + c;
                        StorageType maxVal = input[firstInIdx];

                        for (std::size_t kh = 0; kh < PoolH_; ++kh)
                        {
                            for (std::size_t kw = 0; kw < PoolW_; ++kw)
                            {
                                const std::size_t inIdx =
                                    ((ihStart + kh) * W_ + (iwStart + kw)) *
                                    Channels_ + c;
                                if (input[inIdx] > maxVal)
                                {
                                    maxVal = input[inIdx];
                                }
                            }
                        }

                        output[outPixelOffset + c] = maxVal;
                    }
                }
            }
        }
    };

    /**
     * Average pooling with rounded integer divide. Storage is typically
     * int8_t with int32_t accumulator. AccumType must hold at least
     * PoolH * PoolW * (qmax - qmin) without overflow.
     */
    template<
        typename Storage_,
        typename AccumType_,
        std::size_t H_,
        std::size_t W_,
        std::size_t Channels_,
        std::size_t PoolH_,
        std::size_t PoolW_,
        std::size_t StrideH_ = PoolH_,
        std::size_t StrideW_ = PoolW_>
    struct QAvgPool2D
    {
        typedef Storage_ StorageType;
        typedef AccumType_ AccumulatorType;

        static constexpr std::size_t OutputHeight = (H_ - PoolH_) / StrideH_ + 1;
        static constexpr std::size_t OutputWidth  = (W_ - PoolW_) / StrideW_ + 1;
        static constexpr std::size_t OutputSize   = OutputHeight * OutputWidth * Channels_;
        static constexpr std::size_t InputSize    = H_ * W_ * Channels_;
        static constexpr std::size_t WindowArea   = PoolH_ * PoolW_;

        static_assert(H_ >= PoolH_, "Input height must be >= pool height.");
        static_assert(W_ >= PoolW_, "Input width must be >= pool width.");
        static_assert(StrideH_ > 0 && StrideW_ > 0, "Strides must be > 0.");
        static_assert(PoolH_ > 0 && PoolW_ > 0, "Pool dims must be > 0.");
        static_assert(Channels_ > 0, "Channels must be > 0.");

        StorageType qmin;
        StorageType qmax;

        void forward(const StorageType* input, StorageType* output) const
        {
            for (std::size_t oh = 0; oh < OutputHeight; ++oh)
            {
                const std::size_t ihStart = oh * StrideH_;
                for (std::size_t ow = 0; ow < OutputWidth; ++ow)
                {
                    const std::size_t iwStart = ow * StrideW_;
                    const std::size_t outPixelOffset =
                        (oh * OutputWidth + ow) * Channels_;

                    for (std::size_t c = 0; c < Channels_; ++c)
                    {
                        AccumulatorType sum = static_cast<AccumulatorType>(0);
                        for (std::size_t kh = 0; kh < PoolH_; ++kh)
                        {
                            for (std::size_t kw = 0; kw < PoolW_; ++kw)
                            {
                                const std::size_t inIdx =
                                    ((ihStart + kh) * W_ + (iwStart + kw)) *
                                    Channels_ + c;
                                sum += static_cast<AccumulatorType>(input[inIdx]);
                            }
                        }

                        int32_t avg = detail::roundedDivide(
                            static_cast<int32_t>(sum),
                            static_cast<int32_t>(WindowArea));

                        if (avg < static_cast<int32_t>(qmin)) avg = static_cast<int32_t>(qmin);
                        if (avg > static_cast<int32_t>(qmax)) avg = static_cast<int32_t>(qmax);
                        output[outPixelOffset + c] = static_cast<StorageType>(avg);
                    }
                }
            }
        }
    };

    /**
     * Global average pooling: collapse the H and W axes into a single
     * value per channel. Output is a Channels-length vector that can feed
     * a QDense head, replacing the large flatten-to-dense matrix that
     * dominates flash on small CNN models.
     */
    template<
        typename Storage_,
        typename AccumType_,
        std::size_t H_,
        std::size_t W_,
        std::size_t Channels_>
    struct QGlobalAvgPool2D
    {
        typedef Storage_ StorageType;
        typedef AccumType_ AccumulatorType;

        static constexpr std::size_t OutputSize = Channels_;
        static constexpr std::size_t InputSize  = H_ * W_ * Channels_;
        static constexpr std::size_t WindowArea = H_ * W_;

        static_assert(H_ > 0 && W_ > 0 && Channels_ > 0, "All dims must be > 0.");

        StorageType qmin;
        StorageType qmax;

        void forward(const StorageType* input, StorageType* output) const
        {
            for (std::size_t c = 0; c < Channels_; ++c)
            {
                AccumulatorType sum = static_cast<AccumulatorType>(0);
                for (std::size_t h = 0; h < H_; ++h)
                {
                    for (std::size_t w = 0; w < W_; ++w)
                    {
                        sum += static_cast<AccumulatorType>(
                            input[(h * W_ + w) * Channels_ + c]);
                    }
                }

                int32_t avg = detail::roundedDivide(
                    static_cast<int32_t>(sum),
                    static_cast<int32_t>(WindowArea));

                if (avg < static_cast<int32_t>(qmin)) avg = static_cast<int32_t>(qmin);
                if (avg > static_cast<int32_t>(qmax)) avg = static_cast<int32_t>(qmax);
                output[c] = static_cast<StorageType>(avg);
            }
        }
    };

} // namespace tinymind
