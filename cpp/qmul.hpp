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
 * Quantized elementwise MUL (TFLite MUL semantics).
 *
 * Multiplies two int8 affine tensors and produces an int8 affine output.
 * Required for SE-block / squeeze-excite gating in MobileNet-V3 and
 * EfficientNet, plus general residual gating patterns.
 *
 * Math:
 *
 *   real_y = (scale_A * (qA - zp_A)) * (scale_B * (qB - zp_B))
 *          = (scale_A * scale_B) * (qA - zp_A) * (qB - zp_B)
 *
 *   q_y    = round(real_y / scale_Y) + zp_Y
 *          = MultiplyByQuantizedMultiplier(prod, mult, shift) + zp_Y
 *
 *   prod   = (qA - zp_A) * (qB - zp_B)        // int32, fits with int8 inputs
 *   mult/shift encodes (scale_A * scale_B) / scale_Y as Q0.31.
 *
 * Single Requantizer per layer; the existing apply() does the multiply,
 * shift, zero-point bias, and saturation in one shot.
 *
 * Pure integer at runtime; freestanding-safe.
 */

namespace tinymind {

    template<
        typename InAStorage_,
        typename InBStorage_,
        typename OutStorage_,
        std::size_t N_>
    struct QMul
    {
        typedef InAStorage_ InAType;
        typedef InBStorage_ InBType;
        typedef OutStorage_ OutType;

        static constexpr std::size_t Size = N_;

        InAType input_a_zero_point;
        InBType input_b_zero_point;
        Requantizer<int32_t, OutType> requantizer;

        void forward(const InAType* a, const InBType* b, OutType* out) const
        {
            for (std::size_t i = 0; i < N_; ++i)
            {
                const int32_t a_minus_zp =
                    static_cast<int32_t>(a[i]) -
                    static_cast<int32_t>(input_a_zero_point);
                const int32_t b_minus_zp =
                    static_cast<int32_t>(b[i]) -
                    static_cast<int32_t>(input_b_zero_point);
                const int32_t prod = a_minus_zp * b_minus_zp;
                out[i] = requantizer.apply(prod);
            }
        }
    };

} // namespace tinymind
