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
 * Phase 14 SIMD backend: x86 AVX-VNNI (256-bit VPDPBUSD on Alder Lake+).
 *
 * Gate: TINYMIND_ENABLE_SIMD_AVX_VNNI. Requires SIMD_AVX2.
 *
 * VPDPBUSD computes int32 += sum of 4 pairs of (uint8 * int8) per
 * int32 accumulator lane, with full int32 precision and no
 * intermediate saturation. The input operand must be uint8, so we
 * shift the int8 input by +128 (XOR with 0x80 on the high bit gives
 * the same value as 2's-complement +128):
 *
 *   acc = sum(w * (x + 128 - 128 - zp))
 *       = sum(w * x_u) - (128 + zp) * sum(w)
 *
 * where x_u = (uint8_t)(x + 128). Both VPDPBUSD calls (one with x_u,
 * one with all-ones for sum_w) are bit-exact int32 reductions; the
 * final fold subtracts a single int32 product. Output matches the
 * scalar reference bit-for-bit.
 */

#include "../tinymind_platform.hpp"

#if TINYMIND_ENABLE_SIMD_AVX_VNNI

static_assert(TINYMIND_ENABLE_SIMD_AVX2,
              "TINYMIND_ENABLE_SIMD_AVX_VNNI requires "
              "TINYMIND_ENABLE_SIMD_AVX2. The 256-bit AVX-VNNI form "
              "lives in the AVX path; AVX-512 VNNI is a separate gate.");

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace tinymind { namespace simd { namespace avx_vnni {

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
        __m256i acc_wxu = _mm256_setzero_si256();
        __m256i acc_w   = _mm256_setzero_si256();
        const __m256i bias = _mm256_set1_epi8(static_cast<char>(0x80));
        const __m256i ones_u8 = _mm256_set1_epi8(1);

        std::size_t i = 0;
        for (; i + 32 <= n; i += 32)
        {
            const __m256i xv = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(x + i));
            const __m256i wv = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(w + i));
            // x_u = (uint8_t)(x + 128). Lane-wise int8 addition wraps
            // exactly to the unsigned value we need.
            const __m256i x_u = _mm256_add_epi8(xv, bias);

            // VPDPBUSD: int32 += sum of 4 (uint8 * int8) pairs per lane.
            acc_wxu = _mm256_dpbusd_epi32(acc_wxu, x_u,    wv);
            acc_w   = _mm256_dpbusd_epi32(acc_w,   ones_u8, wv);
        }

        int32_t wxu_sum = horizSum256(acc_wxu);
        int32_t w_sum   = horizSum256(acc_w);

        for (; i < n; ++i)
        {
            // Scalar tail computes the same shifted form to keep the
            // bias bookkeeping identical to the vector path.
            const int32_t x_u_scalar =
                static_cast<int32_t>(static_cast<uint8_t>(
                    static_cast<int32_t>(x[i]) + 128));
            wxu_sum += static_cast<int32_t>(w[i]) * x_u_scalar;
            w_sum   += static_cast<int32_t>(w[i]);
        }

        const int32_t bias_term = (static_cast<int32_t>(128) +
                                   static_cast<int32_t>(zp)) * w_sum;
        return wxu_sum - bias_term;
    }

} } } // namespace tinymind::simd::avx_vnni

#endif // TINYMIND_ENABLE_SIMD_AVX_VNNI
