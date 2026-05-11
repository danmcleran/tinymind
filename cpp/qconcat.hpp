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
 * Quantized 2D channel-axis concatenation (NHWC).
 *
 * Two-input variant: takes two NHWC tensors with matching spatial dims
 * and concatenates them along the channel axis. Each input gets its own
 * Requantizer that rescales (q - zp_input) onto the output affine grid.
 * The "weight scale" passed to buildRequantizer should be 1.0 since
 * concat carries no weights.
 *
 * Used for Inception-style branch merging and any architecture that
 * unions feature maps. Variadic-input concat is deferred; two-input
 * covers the bulk of practical models and stays template-friendly.
 *
 * Tensor layout:
 *
 *   input A   NHWC   [H][W][CA]
 *   input B   NHWC   [H][W][CB]
 *   output    NHWC   [H][W][CA + CB]   A's channels first, B's after
 *
 * Pure integer at runtime; freestanding-safe.
 */

namespace tinymind {

    template<
        typename InAStorage_,
        typename InBStorage_,
        typename OutStorage_,
        std::size_t H_,
        std::size_t W_,
        std::size_t CA_,
        std::size_t CB_>
    struct QConcat2_2D
    {
        typedef InAStorage_ InAType;
        typedef InBStorage_ InBType;
        typedef OutStorage_ OutType;

        static constexpr std::size_t OutputHeight    = H_;
        static constexpr std::size_t OutputWidth     = W_;
        static constexpr std::size_t InputAChannels  = CA_;
        static constexpr std::size_t InputBChannels  = CB_;
        static constexpr std::size_t OutputChannels  = CA_ + CB_;
        static constexpr std::size_t OutputSize      = H_ * W_ * (CA_ + CB_);
        static constexpr std::size_t InputASize      = H_ * W_ * CA_;
        static constexpr std::size_t InputBSize      = H_ * W_ * CB_;

        static_assert(H_ > 0 && W_ > 0, "Spatial dims must be > 0.");
        static_assert(CA_ > 0 && CB_ > 0, "Both inputs must have channels.");

        InAType input_a_zero_point;
        InBType input_b_zero_point;
        Requantizer<int32_t, OutType> requantizer_a;
        Requantizer<int32_t, OutType> requantizer_b;

        void forward(const InAType* a, const InBType* b, OutType* out) const
        {
            for (std::size_t h = 0; h < H_; ++h)
            {
                for (std::size_t w = 0; w < W_; ++w)
                {
                    const std::size_t outPixelOffset =
                        (h * W_ + w) * (CA_ + CB_);
                    const std::size_t aPixelOffset = (h * W_ + w) * CA_;
                    const std::size_t bPixelOffset = (h * W_ + w) * CB_;

                    for (std::size_t c = 0; c < CA_; ++c)
                    {
                        const int32_t v =
                            static_cast<int32_t>(a[aPixelOffset + c]) -
                            static_cast<int32_t>(input_a_zero_point);
                        out[outPixelOffset + c] = requantizer_a.apply(v);
                    }
                    for (std::size_t c = 0; c < CB_; ++c)
                    {
                        const int32_t v =
                            static_cast<int32_t>(b[bPixelOffset + c]) -
                            static_cast<int32_t>(input_b_zero_point);
                        out[outPixelOffset + CA_ + c] = requantizer_b.apply(v);
                    }
                }
            }
        }
    };

} // namespace tinymind
