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
 * Quantized elementwise ADD (TFLite ADD semantics).
 *
 * Adds two int8 affine tensors with potentially different scales and
 * zero_points and produces an int8 affine output. Required for skip
 * connections in ResNet, MobileNet-V2/V3, EfficientNet, U-Net.
 *
 * Math (matches TFLite reference kernel and CMSIS-NN):
 *
 *   twice_max  = 2 * max(scale_A, scale_B)
 *   real_a     = scale_A / twice_max
 *   real_b     = scale_B / twice_max
 *   real_out   = twice_max / ((1 << left_shift) * scale_Y)
 *
 *   shifted_a  = (qA - zp_A) << left_shift
 *   shifted_b  = (qB - zp_B) << left_shift
 *   scaled_a   = MultiplyByQuantizedMultiplier(shifted_a, mult_a, shift_a)
 *   scaled_b   = MultiplyByQuantizedMultiplier(shifted_b, mult_b, shift_b)
 *   sum        = scaled_a + scaled_b
 *   out        = MultiplyByQuantizedMultiplier(sum, mult_out, shift_out)
 *                + zp_Y, saturated to [qmin, qmax]
 *
 * left_shift is normally 20 (TFLite default); it is captured by the
 * (mult_a, shift_a) and (mult_out, shift_out) decompositions so calibration
 * code does the heavy lifting and the runtime only does integer math.
 *
 * Pure integer at runtime; freestanding-safe.
 */

namespace tinymind {

    template<
        typename InAStorage_,
        typename InBStorage_,
        typename OutStorage_,
        std::size_t N_>
    struct QAdd
    {
        typedef InAStorage_ InAType;
        typedef InBStorage_ InBType;
        typedef OutStorage_ OutType;

        static constexpr std::size_t Size = N_;

        InAType input_a_zero_point;
        InBType input_b_zero_point;
        int32_t left_shift;
        int32_t input_a_multiplier;
        int32_t input_a_shift;
        int32_t input_b_multiplier;
        int32_t input_b_shift;
        Requantizer<int32_t, OutType> output_requantizer;

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

                const int32_t shifted_a = a_minus_zp * (static_cast<int32_t>(1) << left_shift);
                const int32_t shifted_b = b_minus_zp * (static_cast<int32_t>(1) << left_shift);

                const int32_t scaled_a = multiplyByQuantizedMultiplier(
                    shifted_a, input_a_multiplier, input_a_shift);
                const int32_t scaled_b = multiplyByQuantizedMultiplier(
                    shifted_b, input_b_multiplier, input_b_shift);

                const int32_t sum = scaled_a + scaled_b;
                out[i] = output_requantizer.apply(sum);
            }
        }
    };

} // namespace tinymind
