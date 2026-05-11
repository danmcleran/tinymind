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
 * Phase 14 SIMD backend: Armv8.1-M MVE-F (float Helium).
 *
 * Gate: TINYMIND_ENABLE_SIMD_HELIUM_MVE_F. M-profile only. Independent
 * of MVE-I per Arm Helium docs: a core can implement either MVE-I or
 * MVE-F alone.
 *
 * Provides a fp32 vector dot product using vfmaq_f32 (Vector Floating
 * point Multiply-Accumulate). The tail is predicated via vctp32q. The
 * accumulator is reduced with vaddvq_f32.
 *
 * Float SIMD reductions are not bit-exact with scalar; the Phase 14
 * bit-exactness invariant applies to integer paths only.
 */

#include "../tinymind_platform.hpp"

#if TINYMIND_ENABLE_SIMD_HELIUM_MVE_F

static_assert(!TINYMIND_ENABLE_SIMD_NEON,
              "TINYMIND_ENABLE_SIMD_HELIUM_MVE_F is M-profile only and is "
              "mutually exclusive with TINYMIND_ENABLE_SIMD_NEON.");
static_assert(!TINYMIND_ENABLE_SIMD_SVE,
              "TINYMIND_ENABLE_SIMD_HELIUM_MVE_F is mutually exclusive with "
              "TINYMIND_ENABLE_SIMD_SVE (SVE is A-profile only).");

#include <cstddef>
#include <arm_mve.h>

namespace tinymind { namespace simd { namespace helium_mve_f {

    inline float floatDotProduct(const float* a, const float* b, std::size_t n)
    {
        float32x4_t acc = vdupq_n_f32(0.0f);
        int32_t remaining = static_cast<int32_t>(n);
        std::size_t i = 0;
        while (remaining > 0)
        {
            const mve_pred16_t pg = vctp32q(static_cast<uint32_t>(remaining));
            const float32x4_t av = vldrwq_z_f32(a + i, pg);
            const float32x4_t bv = vldrwq_z_f32(b + i, pg);
            acc = vfmaq_f32(acc, av, bv);
            i         += 4;
            remaining -= 4;
        }
        return vaddvq_f32(acc);
    }

} } } // namespace tinymind::simd::helium_mve_f

#endif // TINYMIND_ENABLE_SIMD_HELIUM_MVE_F
