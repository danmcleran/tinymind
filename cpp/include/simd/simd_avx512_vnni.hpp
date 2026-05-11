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
 * Phase 14 SIMD backend: x86 AVX-512 VNNI (VPDPBUSD, 512-bit).
 *
 * Gate: TINYMIND_ENABLE_SIMD_AVX512_VNNI. Requires SIMD_AVX512F.
 *
 * Same uint8-shift trick as the AVX-VNNI path, scaled to 512-bit
 * registers (64 int8 inputs per iter). VPDPBUSD on AVX-512 is
 * bit-exact with the scalar reference for the same reason (full int32
 * accumulation, no intermediate saturation).
 */

#include "../tinymind_platform.hpp"

#if TINYMIND_ENABLE_SIMD_AVX512_VNNI

static_assert(TINYMIND_ENABLE_SIMD_AVX512F,
              "TINYMIND_ENABLE_SIMD_AVX512_VNNI requires "
              "TINYMIND_ENABLE_SIMD_AVX512F.");

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace tinymind { namespace simd { namespace avx512_vnni {

    inline int32_t int8DotWithZeroPoint(const int8_t* x, const int8_t* w,
                                        std::size_t n, int8_t zp)
    {
        __m512i acc_wxu = _mm512_setzero_si512();
        __m512i acc_w   = _mm512_setzero_si512();
        const __m512i bias = _mm512_set1_epi8(static_cast<char>(0x80));
        const __m512i ones_u8 = _mm512_set1_epi8(1);

        std::size_t i = 0;
        for (; i + 64 <= n; i += 64)
        {
            const __m512i xv = _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(x + i));
            const __m512i wv = _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(w + i));
            const __m512i x_u = _mm512_add_epi8(xv, bias);

            acc_wxu = _mm512_dpbusd_epi32(acc_wxu, x_u,    wv);
            acc_w   = _mm512_dpbusd_epi32(acc_w,   ones_u8, wv);
        }

        int32_t wxu_sum = _mm512_reduce_add_epi32(acc_wxu);
        int32_t w_sum   = _mm512_reduce_add_epi32(acc_w);

        for (; i < n; ++i)
        {
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

} } } // namespace tinymind::simd::avx512_vnni

#endif // TINYMIND_ENABLE_SIMD_AVX512_VNNI
