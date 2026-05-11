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
 * Phase 14 SIMD dispatch.
 *
 * Exposes the public int8 dot-product primitive used by the quantized
 * layer family:
 *
 *   int32_t tinymind::simd::int8DotWithZeroPoint(
 *       const int8_t* x, const int8_t* w, std::size_t n, int8_t zp)
 *
 *   returns sum_i w[i] * (x[i] - zp).
 *
 * If no TINYMIND_ENABLE_SIMD_* gate is set, the body is the scalar
 * reference; output is byte-identical to the pre-Phase-14 layer code.
 * If a gate is set, the dispatch picks the most capable enabled
 * backend at compile time. Every enabled gate's specialization is
 * bit-exact with the scalar reference (integer-only paths preserve
 * full int32 precision through accumulation).
 *
 * Backend precedence, highest to lowest:
 *   AVX512_VNNI > AVX512F > AVX_VNNI > AVX2 >
 *   NEON_DOTPROD > NEON > SVE > HELIUM_MVE_I > scalar
 *
 * SVE2 has no int8-specific instruction beyond SVE's SDOT; the gate is
 * a forward-compatibility marker for future SVE2-only primitives, so
 * SVE2 paths defer to the SVE backend.
 */

#include "../tinymind_platform.hpp"

#include <cstddef>
#include <cstdint>

#include "simd_neon.hpp"
#include "simd_neon_dotprod.hpp"
#include "simd_neon_fp16.hpp"
#include "simd_sve.hpp"
#include "simd_sve2.hpp"
#include "simd_helium_mve_i.hpp"
#include "simd_helium_mve_f.hpp"
#include "simd_avx2.hpp"
#include "simd_avx_vnni.hpp"
#include "simd_avx512f.hpp"
#include "simd_avx512_vnni.hpp"

namespace tinymind { namespace simd {

    /**
     * Scalar reference implementation. This is the source of truth that
     * every SIMD backend's output must match bit-for-bit. Layer code
     * defers to int8DotWithZeroPoint below rather than calling this
     * directly so the dispatch can pick a faster backend.
     */
    inline int32_t int8DotWithZeroPointScalar(const int8_t* x,
                                              const int8_t* w,
                                              std::size_t n,
                                              int8_t zp)
    {
        int32_t acc = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            acc += static_cast<int32_t>(w[i]) *
                   (static_cast<int32_t>(x[i]) -
                    static_cast<int32_t>(zp));
        }
        return acc;
    }

    inline int32_t int8DotWithZeroPoint(const int8_t* x, const int8_t* w,
                                        std::size_t n, int8_t zp)
    {
#if TINYMIND_ENABLE_SIMD_AVX512_VNNI
        return avx512_vnni::int8DotWithZeroPoint(x, w, n, zp);
#elif TINYMIND_ENABLE_SIMD_AVX512F
        return avx512f::int8DotWithZeroPoint(x, w, n, zp);
#elif TINYMIND_ENABLE_SIMD_AVX_VNNI
        return avx_vnni::int8DotWithZeroPoint(x, w, n, zp);
#elif TINYMIND_ENABLE_SIMD_AVX2
        return avx2::int8DotWithZeroPoint(x, w, n, zp);
#elif TINYMIND_ENABLE_SIMD_NEON_DOTPROD
        return neon_dotprod::int8DotWithZeroPoint(x, w, n, zp);
#elif TINYMIND_ENABLE_SIMD_NEON
        return neon::int8DotWithZeroPoint(x, w, n, zp);
#elif TINYMIND_ENABLE_SIMD_SVE
        return sve::int8DotWithZeroPoint(x, w, n, zp);
#elif TINYMIND_ENABLE_SIMD_HELIUM_MVE_I
        return helium_mve_i::int8DotWithZeroPoint(x, w, n, zp);
#else
        return int8DotWithZeroPointScalar(x, w, n, zp);
#endif
    }

    /**
     * Type-generic dot product with zero-point subtraction. Scalar
     * template body is the source of truth and the path most callers
     * land on; the int8 / int8 / int32 specialization below routes to
     * int8DotWithZeroPoint (which dispatches to the active SIMD
     * backend if any TINYMIND_ENABLE_SIMD_* gate is set, scalar
     * otherwise). Output is byte-identical across the template and the
     * specialization when all gates are off.
     */
    template<typename Input, typename Weight, typename Accum>
    inline Accum dotProductWithZeroPoint(const Input* x, const Weight* w,
                                         std::size_t n, Input zp)
    {
        Accum acc = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const Accum xv = static_cast<Accum>(x[i]) -
                             static_cast<Accum>(zp);
            const Accum wv = static_cast<Accum>(w[i]);
            acc += wv * xv;
        }
        return acc;
    }

    template<>
    inline int32_t dotProductWithZeroPoint<int8_t, int8_t, int32_t>(
        const int8_t* x, const int8_t* w, std::size_t n, int8_t zp)
    {
        return int8DotWithZeroPoint(x, w, n, zp);
    }

    /**
     * Compile-time advertising of which backend the dispatch resolved
     * to. Useful for benchmark reports and the perf_matrix example.
     */
    inline const char* activeBackendName()
    {
#if TINYMIND_ENABLE_SIMD_AVX512_VNNI
        return "avx512_vnni";
#elif TINYMIND_ENABLE_SIMD_AVX512F
        return "avx512f";
#elif TINYMIND_ENABLE_SIMD_AVX_VNNI
        return "avx_vnni";
#elif TINYMIND_ENABLE_SIMD_AVX2
        return "avx2";
#elif TINYMIND_ENABLE_SIMD_NEON_DOTPROD
        return "neon_dotprod";
#elif TINYMIND_ENABLE_SIMD_NEON
        return "neon";
#elif TINYMIND_ENABLE_SIMD_SVE
        return "sve";
#elif TINYMIND_ENABLE_SIMD_HELIUM_MVE_I
        return "helium_mve_i";
#else
        return "scalar";
#endif
    }

} } // namespace tinymind::simd
