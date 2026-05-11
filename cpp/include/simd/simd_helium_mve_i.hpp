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
 * Phase 14 SIMD backend: Armv8.1-M MVE-I (integer Helium).
 *
 * Gate: TINYMIND_ENABLE_SIMD_HELIUM_MVE_I. M-profile only — mutually
 * exclusive with SIMD_NEON / SIMD_SVE (application-class A/R profile).
 *
 * Provides an int8 inner product using vmladavaq_s8 (Vector Multiply
 * Add Dot product Across Vector Accumulate, Signed 8-bit), which
 * reduces a pair of int8x16 vectors into a scalar int32 accumulator in
 * a single instruction. The tail is handled with a predicated load
 * (vldrbq_z_s8) that zeroes inactive lanes; zero contributions do not
 * affect the dot product.
 *
 * Bit-exact with the scalar reference for the same reason as the other
 * integer backends.
 */

#include "../tinymind_platform.hpp"

#if TINYMIND_ENABLE_SIMD_HELIUM_MVE_I

static_assert(!TINYMIND_ENABLE_SIMD_NEON,
              "TINYMIND_ENABLE_SIMD_HELIUM_MVE_I is M-profile only and is "
              "mutually exclusive with TINYMIND_ENABLE_SIMD_NEON (Helium "
              "ships on Cortex-M55 / M85 / M52, never alongside NEON).");
static_assert(!TINYMIND_ENABLE_SIMD_SVE,
              "TINYMIND_ENABLE_SIMD_HELIUM_MVE_I is mutually exclusive with "
              "TINYMIND_ENABLE_SIMD_SVE (SVE is A-profile only).");

#include <cstddef>
#include <cstdint>
#include <arm_mve.h>

namespace tinymind { namespace simd { namespace helium_mve_i {

    inline int32_t int8DotWithZeroPoint(const int8_t* x, const int8_t* w,
                                        std::size_t n, int8_t zp)
    {
        int32_t acc_wx = 0;
        int32_t acc_w  = 0;
        int32_t remaining = static_cast<int32_t>(n);
        std::size_t i = 0;
        const int8x16_t ones = vdupq_n_s8(static_cast<int8_t>(1));

        while (remaining > 0)
        {
            const mve_pred16_t pg = vctp8q(static_cast<uint32_t>(remaining));
            const int8x16_t xv = vldrbq_z_s8(x + i, pg);
            const int8x16_t wv = vldrbq_z_s8(w + i, pg);
            // Inactive lanes are zeroed by the predicated load, so the
            // unpredicated dot product across all 16 lanes contributes
            // only the active lanes.
            acc_wx = vmladavaq_s8(acc_wx, wv, xv);
            acc_w  = vmladavaq_s8(acc_w,  wv, ones);
            i         += 16;
            remaining -= 16;
        }

        return acc_wx - static_cast<int32_t>(zp) * acc_w;
    }

} } } // namespace tinymind::simd::helium_mve_i

#endif // TINYMIND_ENABLE_SIMD_HELIUM_MVE_I
