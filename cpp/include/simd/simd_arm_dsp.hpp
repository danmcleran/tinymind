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
 * Phase 14 SIMD backend: Arm DSP extension (32-bit packed SIMD).
 *
 * Gate: TINYMIND_ENABLE_SIMD_ARM_DSP. Targets the AArch32 DSP extension
 * that ships on Cortex-M4 / M7 / M33 / M35P / M55 / M85 (Armv7E-M and
 * Armv8-M Mainline with DSP). Cortex-M0 / M0+ / M3 / M23 lack it, and
 * AArch64 has no equivalent instructions, so the header refuses to
 * compile unless the toolchain advertises __ARM_FEATURE_DSP.
 *
 * Provides an int8 inner product built on SMLAD (Signed Multiply
 * Accumulate Dual), which does two 16x16 multiplies and adds both
 * products to a 32-bit accumulator in one single-cycle instruction on
 * Cortex-M4. Four int8 values are fetched with one 32-bit load and
 * widened with SXTB16, which sign-extends bytes 0 and 2 into two int16
 * lanes; rotating the word by 8 first selects bytes 1 and 3. The input
 * zero point is subtracted in the int16 lanes with SSUB16 (x - zp lies
 * in [-255, 255], so no lane can wrap). Weights and inputs are unpacked
 * with the same lane selection, so the byte pairing is independent of
 * load endianness. A scalar loop handles the n % 4 tail.
 *
 * Bit-exact with the scalar reference: every product and partial sum is
 * the same int32 quantity, only the order of the additions differs.
 */

#include "../tinymind_platform.hpp"

#if TINYMIND_ENABLE_SIMD_ARM_DSP

// The prerequisite here is a toolchain target, not another gate, so it is
// checked against the ACLE feature macro rather than with a static_assert.
#if !defined(__ARM_FEATURE_DSP)
#error "TINYMIND_ENABLE_SIMD_ARM_DSP requires a core with the Arm DSP extension (__ARM_FEATURE_DSP): Cortex-M4 / M7 / M33 / M35P / M55 / M85. Cortex-M0 / M0+ / M3 / M23 and AArch64 targets do not have SMLAD; leave the gate off for them."
#else

#include <cstddef>
#include <cstdint>
#include <arm_acle.h>

namespace tinymind { namespace simd { namespace arm_dsp {

    // One 32-bit load of four int8 values. __builtin_memcpy keeps it free
    // of aliasing and alignment UB while compiling to a single LDR wherever
    // the target permits unaligned word loads (__ARM_FEATURE_UNALIGNED).
    // GCC enables that by default on Armv7E-M / Armv8-M Mainline; Clang's
    // bare-metal driver does not, so Clang builds want -munaligned-access or
    // every load here splits into four LDRBs (correct, but slower).
    inline int8x4_t load4(const int8_t* p)
    {
        int8x4_t v;
        __builtin_memcpy(&v, p, sizeof(v));
        return v;
    }

    // SXTB16 of the word rotated right by 8, i.e. bytes 1 and 3. The
    // instruction takes the rotate as a free operand. Clang folds a plain
    // rotate expression into it; GCC 13 does not (and its arm_acle.h has
    // no __ror), so GCC gets the encoding directly.
    inline int16x2_t sxtb16Ror8(int8x4_t v)
    {
#if defined(__clang__)
        const uint32_t u = static_cast<uint32_t>(v);
        return __sxtb16(static_cast<int8x4_t>((u >> 8) | (u << 24)));
#else
        int16x2_t r;
        __asm__("sxtb16 %0, %1, ror #8" : "=r"(r) : "r"(v));
        return r;
#endif
    }

    inline int32_t int8DotWithZeroPoint(const int8_t* x, const int8_t* w,
                                        std::size_t n, int8_t zp)
    {
        // zp replicated into both int16 lanes, as SSUB16 expects.
        const uint32_t zp16 = static_cast<uint16_t>(static_cast<int16_t>(zp));
        const int16x2_t zpPair = static_cast<int16x2_t>(zp16 | (zp16 << 16));

        int32_t acc = 0;

        for (std::size_t blocks = n >> 2; blocks != 0; --blocks)
        {
            const int8x4_t xv = load4(x);
            const int8x4_t wv = load4(w);
            x += 4;
            w += 4;

            const int16x2_t xEven = __ssub16(__sxtb16(xv), zpPair);
            const int16x2_t xOdd  = __ssub16(sxtb16Ror8(xv), zpPair);
            const int16x2_t wEven = __sxtb16(wv);
            const int16x2_t wOdd  = sxtb16Ror8(wv);

            acc = __smlad(wEven, xEven, acc);
            acc = __smlad(wOdd,  xOdd,  acc);
        }

        for (std::size_t i = 0; i < (n & 3u); ++i)
        {
            acc += static_cast<int32_t>(w[i]) *
                   (static_cast<int32_t>(x[i]) - static_cast<int32_t>(zp));
        }

        return acc;
    }

} } } // namespace tinymind::simd::arm_dsp

#endif // __ARM_FEATURE_DSP

#endif // TINYMIND_ENABLE_SIMD_ARM_DSP
