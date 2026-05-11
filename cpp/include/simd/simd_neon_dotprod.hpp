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
 * Phase 14 SIMD backend: Armv8.2-A FEAT_DotProd (SDOT / UDOT).
 *
 * Gate: TINYMIND_ENABLE_SIMD_NEON_DOTPROD. Requires SIMD_NEON.
 *
 * SDOT performs a four-way signed int8 dot product into each lane of an
 * int32x4 accumulator, returning four int32 partial sums per
 * instruction over 16 int8 inputs. This is the headline int8 MAC
 * accelerator on modern Arm parts (Cortex-A55, A510, A715, Neoverse,
 * optionally Cortex-R82).
 *
 * Result is bit-exact with the scalar reference for the same reason as
 * the NEON baseline path: every partial product fits in int16, every
 * accumulation step is exact int32.
 */

#include "../tinymind_platform.hpp"

#if TINYMIND_ENABLE_SIMD_NEON_DOTPROD

static_assert(TINYMIND_ENABLE_SIMD_NEON,
              "TINYMIND_ENABLE_SIMD_NEON_DOTPROD requires "
              "TINYMIND_ENABLE_SIMD_NEON. Per Arm Advanced SIMD reference, "
              "SDOT/UDOT live inside the Advanced SIMD instruction set; "
              "there is no DotProd without NEON.");

#include <cstddef>
#include <cstdint>
#include <arm_neon.h>

namespace tinymind { namespace simd { namespace neon_dotprod {

    inline int32_t int8DotWithZeroPoint(const int8_t* x, const int8_t* w,
                                        std::size_t n, int8_t zp)
    {
        int32x4_t acc_wx = vdupq_n_s32(0);
        int32x4_t acc_w  = vdupq_n_s32(0);
        const int8x16_t ones = vdupq_n_s8(1);

        std::size_t i = 0;
        for (; i + 16 <= n; i += 16)
        {
            const int8x16_t xv = vld1q_s8(x + i);
            const int8x16_t wv = vld1q_s8(w + i);
            acc_wx = vdotq_s32(acc_wx, wv, xv);
            acc_w  = vdotq_s32(acc_w,  wv, ones);
        }

        int32_t wx_sum = vaddvq_s32(acc_wx);
        int32_t w_sum  = vaddvq_s32(acc_w);

        for (; i < n; ++i)
        {
            wx_sum += static_cast<int32_t>(w[i]) * static_cast<int32_t>(x[i]);
            w_sum  += static_cast<int32_t>(w[i]);
        }

        return wx_sum - static_cast<int32_t>(zp) * w_sum;
    }

} } } // namespace tinymind::simd::neon_dotprod

#endif // TINYMIND_ENABLE_SIMD_NEON_DOTPROD
