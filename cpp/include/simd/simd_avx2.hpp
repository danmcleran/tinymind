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
 * Phase 14 SIMD backend: x86 AVX2.
 *
 * Gate: TINYMIND_ENABLE_SIMD_AVX2.
 *
 * Uses widen-to-int16 + _mm256_madd_epi16 for the int8 inner product:
 *
 *   load 16 int8 from each operand
 *   widen to int16 via _mm256_cvtepi8_epi16
 *   _mm256_madd_epi16 computes 8 int32 lanes, each the sum of two int32
 *     products of int16 inputs (no saturation risk)
 *   accumulate into a running int32 sum
 *
 * Avoids _mm256_maddubs_epi16 deliberately: PMADDUBSW saturates at
 * int16 on the pair-sum step, which is reachable with our int8 input
 * range and would break bit-exactness vs the scalar reference. The
 * cvtepi8_epi16 + madd_epi16 chain costs one extra cycle per iter but
 * is exact.
 *
 * Result is bit-exact with the scalar reference: every int32 product
 * is exact, every accumulation step preserves int32 precision.
 */

#include "../tinymind_platform.hpp"

#if TINYMIND_ENABLE_SIMD_AVX2

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace tinymind { namespace simd { namespace avx2 {

    inline int32_t horizSum256(__m256i v)
    {
        const __m128i lo = _mm256_castsi256_si128(v);
        const __m128i hi = _mm256_extracti128_si256(v, 1);
        __m128i s = _mm_add_epi32(lo, hi);
        s = _mm_hadd_epi32(s, s);
        s = _mm_hadd_epi32(s, s);
        return _mm_cvtsi128_si32(s);
    }

    inline int32_t int8DotWithZeroPoint(const int8_t* x, const int8_t* w,
                                        std::size_t n, int8_t zp)
    {
        __m256i acc_wx = _mm256_setzero_si256();
        __m256i acc_w  = _mm256_setzero_si256();
        const __m256i ones16 = _mm256_set1_epi16(1);

        std::size_t i = 0;
        for (; i + 16 <= n; i += 16)
        {
            const __m128i x8 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(x + i));
            const __m128i w8 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(w + i));
            const __m256i x16 = _mm256_cvtepi8_epi16(x8);
            const __m256i w16 = _mm256_cvtepi8_epi16(w8);

            // _mm256_madd_epi16 returns 8 int32 lanes; each lane is
            // (a0*b0 + a1*b1) with full int32 precision.
            const __m256i wx32 = _mm256_madd_epi16(x16, w16);
            acc_wx = _mm256_add_epi32(acc_wx, wx32);

            // Sum of weights, in int32 lanes.
            const __m256i w32 = _mm256_madd_epi16(w16, ones16);
            acc_w = _mm256_add_epi32(acc_w, w32);
        }

        int32_t wx_sum = horizSum256(acc_wx);
        int32_t w_sum  = horizSum256(acc_w);

        for (; i < n; ++i)
        {
            wx_sum += static_cast<int32_t>(w[i]) * static_cast<int32_t>(x[i]);
            w_sum  += static_cast<int32_t>(w[i]);
        }

        return wx_sum - static_cast<int32_t>(zp) * w_sum;
    }

} } } // namespace tinymind::simd::avx2

#endif // TINYMIND_ENABLE_SIMD_AVX2
