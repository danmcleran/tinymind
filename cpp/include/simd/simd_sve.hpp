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
 * Phase 14 SIMD backend: Scalable Vector Extension (SVE).
 *
 * Gate: TINYMIND_ENABLE_SIMD_SVE. Requires SIMD_NEON per GCC AArch64
 * docs ("+sve also enables Advanced SIMD and floating-point
 * instructions"). The vector length is unknown at compile time, so the
 * loop uses svwhilelt_b8 + svcntb to step through the input in
 * hardware-vector-length-sized chunks; the same source compiles for any
 * SVE-enabled core (Neoverse V1 = 256-bit, V2 = 128-bit / 256-bit, A64FX
 * = 512-bit, ...).
 *
 * Result is bit-exact with the scalar reference. SVDOT reduces four
 * int8 lanes per int32 accumulator slot; the final svaddv reduction
 * sums the accumulator vector deterministically.
 */

#include "../tinymind_platform.hpp"

#if TINYMIND_ENABLE_SIMD_SVE

static_assert(TINYMIND_ENABLE_SIMD_NEON,
              "TINYMIND_ENABLE_SIMD_SVE requires TINYMIND_ENABLE_SIMD_NEON. "
              "Per GCC AArch64 options, `+sve also enables Advanced SIMD "
              "and floating-point instructions` — SVE always implies NEON.");

#include <cstddef>
#include <cstdint>
#include <arm_sve.h>

namespace tinymind { namespace simd { namespace sve {

    inline int32_t int8DotWithZeroPoint(const int8_t* x, const int8_t* w,
                                        std::size_t n, int8_t zp)
    {
        svint32_t acc_wx = svdup_n_s32(0);
        svint32_t acc_w  = svdup_n_s32(0);

        std::size_t i = 0;
        while (i < n)
        {
            const svbool_t pg = svwhilelt_b8(i, n);
            const svint8_t xv = svld1_s8(pg, x + i);
            const svint8_t wv = svld1_s8(pg, w + i);
            // Predicated-zero lanes already contribute 0 * 0 = 0, so the
            // dot product across the full vector matches the scalar
            // partial sum exactly.
            const svint8_t ones = svdup_n_s8_z(pg, 1);
            acc_wx = svdot_s32(acc_wx, wv, xv);
            acc_w  = svdot_s32(acc_w,  wv, ones);
            i += svcntb();
        }

        const svbool_t all_lanes = svptrue_b32();
        const int32_t wx_sum = svaddv_s32(all_lanes, acc_wx);
        const int32_t w_sum  = svaddv_s32(all_lanes, acc_w);
        return wx_sum - static_cast<int32_t>(zp) * w_sum;
    }

} } } // namespace tinymind::simd::sve

#endif // TINYMIND_ENABLE_SIMD_SVE
