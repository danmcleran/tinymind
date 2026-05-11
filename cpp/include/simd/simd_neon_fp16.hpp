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
 * Phase 14 SIMD backend: Armv8.2-A FEAT_FP16 (half-precision vector).
 *
 * Gate: TINYMIND_ENABLE_SIMD_NEON_FP16. Requires SIMD_NEON. Pairs with
 * the Phase 9 fp16_t storage tier from tinymind_fp16.hpp.
 *
 * Provides an fp16 vector dot product using the float16x8 fused
 * multiply-add intrinsic vfmaq_f16. The accumulator is upcast to
 * float32 at the end for a stable horizontal reduction. The lane
 * arithmetic itself is fp16-precision; callers that need fp32-precision
 * accumulation should widen at load time and use a scalar path.
 *
 * Float SIMD reductions are not bit-exact with scalar (order of
 * addition affects rounding); the Phase 14 invariant of bit-exactness
 * applies only to the integer dot product primitives. fp16 paths are
 * a perf shortcut for already-fp16 callers.
 */

#include "../tinymind_platform.hpp"

#if TINYMIND_ENABLE_SIMD_NEON_FP16

static_assert(TINYMIND_ENABLE_SIMD_NEON,
              "TINYMIND_ENABLE_SIMD_NEON_FP16 requires "
              "TINYMIND_ENABLE_SIMD_NEON. Per Arm intrinsics reference, "
              "the FEAT_FP16 vector forms live inside the Advanced SIMD "
              "instruction set; there is no fp16 vector arithmetic "
              "without NEON.");

#include <cstddef>
#include <arm_neon.h>

namespace tinymind { namespace simd { namespace neon_fp16 {

    inline float fp16DotProduct(const __fp16* a, const __fp16* b, std::size_t n)
    {
        float16x8_t acc = vdupq_n_f16(static_cast<__fp16>(0.0f));
        std::size_t i = 0;
        for (; i + 8 <= n; i += 8)
        {
            const float16x8_t av = vld1q_f16(a + i);
            const float16x8_t bv = vld1q_f16(b + i);
            acc = vfmaq_f16(acc, av, bv);
        }
        const float32x4_t lo = vcvt_f32_f16(vget_low_f16(acc));
        const float32x4_t hi = vcvt_f32_f16(vget_high_f16(acc));
        float32x4_t sum32 = vaddq_f32(lo, hi);
        float result = vaddvq_f32(sum32);
        for (; i < n; ++i)
        {
            result += static_cast<float>(a[i]) * static_cast<float>(b[i]);
        }
        return result;
    }

} } } // namespace tinymind::simd::neon_fp16

#endif // TINYMIND_ENABLE_SIMD_NEON_FP16
