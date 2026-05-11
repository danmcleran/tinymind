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

/*
 * Phase 14 SIMD backend: Armv7 / Armv8-A Advanced SIMD (NEON) baseline.
 *
 * Gate: TINYMIND_ENABLE_SIMD_NEON.
 *
 * Provides an int8 inner-product primitive that uses VMULL (widening
 * multiply) and VPADAL (pairwise widening accumulate) to reduce a pair
 * of int8 vectors into an int32 dot product. This is the baseline path
 * for Arm parts that ship NEON but not the Armv8.2-A FEAT_DotProd
 * extension; if SDOT is available, simd_neon_dotprod.hpp supersedes
 * this one.
 *
 * The reduction is bit-exact with the scalar reference. Each int8 *
 * int8 product fits in int16, the pairwise widening accumulate sums
 * into int32 in a different lane order than scalar but the sum of the
 * same finite int32 values is associative and produces the same
 * result.
 *
 * Header is intrinsic-only and pulls in <arm_neon.h>; do not include
 * it from freestanding paths where NEON is not present.
 */

#include "../tinymind_platform.hpp"

#include <cstddef>
#include <cstdint>

#if TINYMIND_ENABLE_SIMD_NEON

#include <arm_neon.h>

namespace tinymind { namespace simd { namespace neon {

    inline int32_t horizSum32(int32x4_t v)
    {
        // AArch32-compatible horizontal sum: pairwise reduce twice.
        int32x2_t s = vadd_s32(vget_low_s32(v), vget_high_s32(v));
        return vget_lane_s32(vpadd_s32(s, s), 0);
    }

    /**
     * int8 dot product with input zero-point subtraction:
     *
     *   acc = sum_i w[i] * (x[i] - zp)
     *       = sum_i (w[i] * x[i]) - zp * sum_i w[i]
     *
     * Computes both partial sums in parallel int32 vectors, then folds.
     * Result is bit-exact with the scalar reference because every
     * partial product fits in int16 and every accumulation step
     * preserves int32 precision.
     */
    inline int32_t int8DotWithZeroPoint(const int8_t* x, const int8_t* w,
                                        std::size_t n, int8_t zp)
    {
        int32x4_t acc_wx = vdupq_n_s32(0);
        int32x4_t acc_w  = vdupq_n_s32(0);

        std::size_t i = 0;
        for (; i + 16 <= n; i += 16)
        {
            const int8x16_t xv = vld1q_s8(x + i);
            const int8x16_t wv = vld1q_s8(w + i);

            // 8-lane low/high widening multiplies into int16.
            const int16x8_t lo = vmull_s8(vget_low_s8(xv),  vget_low_s8(wv));
            const int16x8_t hi = vmull_s8(vget_high_s8(xv), vget_high_s8(wv));
            acc_wx = vpadalq_s16(acc_wx, lo);
            acc_wx = vpadalq_s16(acc_wx, hi);

            const int16x8_t wlo = vmovl_s8(vget_low_s8(wv));
            const int16x8_t whi = vmovl_s8(vget_high_s8(wv));
            acc_w = vpadalq_s16(acc_w, wlo);
            acc_w = vpadalq_s16(acc_w, whi);
        }

        int32_t wx_sum = horizSum32(acc_wx);
        int32_t w_sum  = horizSum32(acc_w);

        for (; i < n; ++i)
        {
            wx_sum += static_cast<int32_t>(w[i]) * static_cast<int32_t>(x[i]);
            w_sum  += static_cast<int32_t>(w[i]);
        }

        return wx_sum - static_cast<int32_t>(zp) * w_sum;
    }

} } } // namespace tinymind::simd::neon

#endif // TINYMIND_ENABLE_SIMD_NEON
