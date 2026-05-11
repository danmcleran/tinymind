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
 * Phase 14 SIMD backend: x86 AVX-512 Foundation (AVX-512F + BW).
 *
 * Gate: TINYMIND_ENABLE_SIMD_AVX512F.
 *
 * Same widen-to-int16 + _mm512_madd_epi16 chain as the AVX2 backend,
 * doubled to 512-bit registers. Processes 32 int8 inputs per iteration
 * (vs 16 for AVX2). Bit-exact with scalar.
 *
 * Note: _mm512_cvtepi8_epi16 and _mm512_madd_epi16 are AVX-512BW
 * instructions, which ship together with AVX-512F on every shipping
 * AVX-512 CPU (Skylake-X / Ice Lake / Tiger Lake / Sapphire Rapids).
 * The library treats AVX-512F as implying AVX-512BW for this path.
 */

#include "../tinymind_platform.hpp"

#if TINYMIND_ENABLE_SIMD_AVX512F

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace tinymind { namespace simd { namespace avx512f {

    inline int32_t int8DotWithZeroPoint(const int8_t* x, const int8_t* w,
                                        std::size_t n, int8_t zp)
    {
        __m512i acc_wx = _mm512_setzero_si512();
        __m512i acc_w  = _mm512_setzero_si512();
        const __m512i ones16 = _mm512_set1_epi16(1);

        std::size_t i = 0;
        for (; i + 32 <= n; i += 32)
        {
            const __m256i x8 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(x + i));
            const __m256i w8 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(w + i));
            const __m512i x16 = _mm512_cvtepi8_epi16(x8);
            const __m512i w16 = _mm512_cvtepi8_epi16(w8);

            const __m512i wx32 = _mm512_madd_epi16(x16, w16);
            acc_wx = _mm512_add_epi32(acc_wx, wx32);

            const __m512i w32 = _mm512_madd_epi16(w16, ones16);
            acc_w = _mm512_add_epi32(acc_w, w32);
        }

        int32_t wx_sum = _mm512_reduce_add_epi32(acc_wx);
        int32_t w_sum  = _mm512_reduce_add_epi32(acc_w);

        for (; i < n; ++i)
        {
            wx_sum += static_cast<int32_t>(w[i]) * static_cast<int32_t>(x[i]);
            w_sum  += static_cast<int32_t>(w[i]);
        }

        return wx_sum - static_cast<int32_t>(zp) * w_sum;
    }

} } } // namespace tinymind::simd::avx512f

#endif // TINYMIND_ENABLE_SIMD_AVX512F
