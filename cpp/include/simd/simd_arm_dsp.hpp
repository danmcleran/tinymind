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
 * zero point is folded into the widening of x with SXTAB16, which adds
 * a packed (-zp, -zp) pair while it sign-extends (x - zp lies in
 * [-255, 255], so no lane can wrap). Weights and inputs are unpacked
 * with the same lane selection, so the byte pairing is independent of
 * load endianness. A scalar loop handles the n % 4 tail.
 *
 * The block loop is unrolled with a pragma rather than left to the
 * optimizer. Layers call this with a compile-time length, and at -O2 GCC
 * otherwise keeps the loop rolled inside the layer's output loop, where
 * the requantizer's hoisted constants leave it too few registers: the
 * loaded words were spilled to the stack on every iteration, and the
 * cost swung by up to 1.7x with unrelated surrounding code. Unrolled, the
 * loop needs no loop-carried pointers and the -O2 build matches -O3. The
 * pragma is skipped at -Os/-Oz, where Clang would otherwise honor it.
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

    // SXTAB16 of the word rotated right by 8: a + (bytes 1 and 3 sign-extended),
    // lane by lane. Same GCC/Clang split as sxtb16Ror8.
    inline int16x2_t sxtab16Ror8(int16x2_t a, int8x4_t v)
    {
#if defined(__clang__)
        const uint32_t u = static_cast<uint32_t>(v);
        return __sxtab16(a, static_cast<int8x4_t>((u >> 8) | (u << 24)));
#else
        int16x2_t r;
        __asm__("sxtab16 %0, %1, %2, ror #8" : "=r"(r) : "r"(a), "r"(v));
        return r;
#endif
    }

    inline int32_t int8DotWithZeroPoint(const int8_t* x, const int8_t* w,
                                        std::size_t n, int8_t zp)
    {
        // -zp replicated into both int16 lanes. SXTAB16 adds it while it
        // sign-extends, so x - zp costs no extra instruction.
        const uint32_t nzp16 = static_cast<uint16_t>(static_cast<int16_t>(-static_cast<int16_t>(zp)));
        const int16x2_t negZpPair = static_cast<int16x2_t>(nzp16 | (nzp16 << 16));

        int32_t acc = 0;

#if !defined(__OPTIMIZE_SIZE__)
#pragma GCC unroll 16
#endif
        for (std::size_t blocks = n >> 2; blocks != 0; --blocks)
        {
            const int8x4_t xv = load4(x);
            const int8x4_t wv = load4(w);
            x += 4;
            w += 4;

            acc = __smlad(__sxtb16(wv), __sxtab16(negZpPair, xv), acc);
            acc = __smlad(sxtb16Ror8(wv), sxtab16Ror8(negZpPair, xv), acc);
        }

        for (std::size_t i = 0; i < (n & 3u); ++i)
        {
            acc += static_cast<int32_t>(w[i]) *
                   (static_cast<int32_t>(x[i]) - static_cast<int32_t>(zp));
        }

        return acc;
    }

    /**
     * Two inputs against two weight rows, n elements each:
     *   acc[0] += w0 . (x0 - zp)    acc[1] += w1 . (x0 - zp)
     *   acc[2] += w0 . (x1 - zp)    acc[3] += w1 . (x1 - zp)
     * Each widened input pair feeds two SMLADs and each widened weight pair
     * two more, so a block of four elements is 4 loads, 8 extends and 8
     * SMLADs for 16 multiply-accumulates, against 2 loads, 4 extends and 4
     * SMLADs for 4 in int8DotWithZeroPoint. Convolutions call it for two
     * output pixels times two filters. Same products, so bit-exact.
     */
    inline void int8Dot2x2WithZeroPoint(const int8_t* x0, const int8_t* x1,
                                        const int8_t* w0, const int8_t* w1,
                                        std::size_t n, int8_t zp, int32_t* acc)
    {
        const uint32_t nzp16 = static_cast<uint16_t>(static_cast<int16_t>(-static_cast<int16_t>(zp)));
        const int16x2_t negZpPair = static_cast<int16x2_t>(nzp16 | (nzp16 << 16));

        int32_t a00 = acc[0];
        int32_t a10 = acc[1];
        int32_t a01 = acc[2];
        int32_t a11 = acc[3];

        // The second input and weight rows are addressed from the first. A
        // caller that pairs neighbouring pixels and filters passes rows a
        // compile-time distance apart, and after inlining these become
        // immediate offsets: two pointers advance instead of four.
        const std::ptrdiff_t dx = x1 - x0;
        const std::ptrdiff_t dw = w1 - w0;

#if !defined(__OPTIMIZE_SIZE__)
#pragma GCC unroll 8
#endif
        for (std::size_t blocks = n >> 2; blocks != 0; --blocks)
        {
            // Inputs first, widened once and held; then each weight word is
            // widened and spent on both pixels before the next is loaded, so
            // at most one widened weight is live.
            const int8x4_t x0v = load4(x0);
            const int16x2_t x0e = __sxtab16(negZpPair, x0v);
            const int16x2_t x0o = sxtab16Ror8(negZpPair, x0v);
            const int8x4_t x1v = load4(x0 + dx);
            const int16x2_t x1e = __sxtab16(negZpPair, x1v);
            const int16x2_t x1o = sxtab16Ror8(negZpPair, x1v);
            x0 += 4;

            const int8x4_t w0v = load4(w0);
            const int16x2_t w0e = __sxtb16(w0v);
            a00 = __smlad(w0e, x0e, a00);
            a01 = __smlad(w0e, x1e, a01);
            const int16x2_t w0o = sxtb16Ror8(w0v);
            a00 = __smlad(w0o, x0o, a00);
            a01 = __smlad(w0o, x1o, a01);

            const int8x4_t w1v = load4(w0 + dw);
            const int16x2_t w1e = __sxtb16(w1v);
            a10 = __smlad(w1e, x0e, a10);
            a11 = __smlad(w1e, x1e, a11);
            const int16x2_t w1o = sxtb16Ror8(w1v);
            a10 = __smlad(w1o, x0o, a10);
            a11 = __smlad(w1o, x1o, a11);
            w0 += 4;
        }

        for (std::size_t i = 0; i < (n & 3u); ++i)
        {
            const int32_t xa = static_cast<int32_t>(x0[i]) - static_cast<int32_t>(zp);
            const int32_t xb = static_cast<int32_t>(x0[dx + static_cast<std::ptrdiff_t>(i)]) -
                               static_cast<int32_t>(zp);
            const int32_t wa = static_cast<int32_t>(w0[i]);
            const int32_t wb = static_cast<int32_t>(w0[dw + static_cast<std::ptrdiff_t>(i)]);
            a00 += wa * xa;
            a10 += wb * xa;
            a01 += wa * xb;
            a11 += wb * xb;
        }

        acc[0] = a00;
        acc[1] = a10;
        acc[2] = a01;
        acc[3] = a11;
    }

    /**
     * A convolution's 2x2 block over all KH kernel rows: two output pixels
     * PixelStep elements apart times two filters WindowLength apart, each
     * kernel row RowLength elements, accumulated into acc as in
     * int8Dot2x2WithZeroPoint. Kept out of line on purpose. Inlined into a
     * layer's loops, the four accumulators, eight widened values and two
     * base pointers compete with the caller's live state for the 14
     * allocatable registers, and whether they spilled swung the layer's
     * cost by 30% with unrelated changes to the surrounding code. Out of
     * line, the kernel owns the register file; the call costs a few cycles
     * against KH * RowLength / 4 blocks of work. The offsets are template
     * parameters, so every load is base plus immediate.
     */
    template<std::size_t KH, std::size_t RowLength, std::size_t PixelStep, std::size_t WindowLength>
    __attribute__((noinline)) void int8ConvRows2x2WithZeroPoint(const int8_t* const* rows, std::size_t offset,
                                                               const int8_t* w0, int8_t zp, int32_t* acc)
    {
        const uint32_t nzp16 = static_cast<uint16_t>(static_cast<int16_t>(-static_cast<int16_t>(zp)));
        const int16x2_t negZpPair = static_cast<int16x2_t>(nzp16 | (nzp16 << 16));

        int32_t a00 = acc[0];
        int32_t a10 = acc[1];
        int32_t a01 = acc[2];
        int32_t a11 = acc[3];

        for (std::size_t kh = 0; kh < KH; ++kh)
        {
            const int8_t* x = rows[kh] + offset;
            const int8_t* w = w0 + kh * RowLength;

            // Two blocks per iteration, measured on a Cortex-M4 (F411, GCC 13,
            // -O3) over the MLPerf Tiny ResNet-8: the loop then keeps its
            // accumulators, widened values and pointers in registers with no
            // spill, in about 160 bytes of code. Three, four and sixteen
            // blocks made the whole model 22-27% slower, one block 15%.
#if !defined(__OPTIMIZE_SIZE__)
#pragma GCC unroll 2
#endif
            for (std::size_t blocks = RowLength >> 2; blocks != 0; --blocks)
            {
                const int8x4_t x0v = load4(x);
                const int16x2_t x0e = __sxtab16(negZpPair, x0v);
                const int16x2_t x0o = sxtab16Ror8(negZpPair, x0v);
                const int8x4_t x1v = load4(x + PixelStep);
                const int16x2_t x1e = __sxtab16(negZpPair, x1v);
                const int16x2_t x1o = sxtab16Ror8(negZpPair, x1v);
                x += 4;

                const int8x4_t w0v = load4(w);
                const int16x2_t w0e = __sxtb16(w0v);
                a00 = __smlad(w0e, x0e, a00);
                a01 = __smlad(w0e, x1e, a01);
                const int16x2_t w0o = sxtb16Ror8(w0v);
                a00 = __smlad(w0o, x0o, a00);
                a01 = __smlad(w0o, x1o, a01);

                const int8x4_t w1v = load4(w + WindowLength);
                const int16x2_t w1e = __sxtb16(w1v);
                a10 = __smlad(w1e, x0e, a10);
                a11 = __smlad(w1e, x1e, a11);
                const int16x2_t w1o = sxtb16Ror8(w1v);
                a10 = __smlad(w1o, x0o, a10);
                a11 = __smlad(w1o, x1o, a11);
                w += 4;
            }

            for (std::size_t i = 0; i < (RowLength & 3u); ++i)
            {
                const int32_t xa = static_cast<int32_t>(x[i]) - static_cast<int32_t>(zp);
                const int32_t xb = static_cast<int32_t>(x[PixelStep + i]) - static_cast<int32_t>(zp);
                const int32_t wa = static_cast<int32_t>(w[i]);
                const int32_t wb = static_cast<int32_t>(w[WindowLength + i]);
                a00 += wa * xa;
                a10 += wb * xa;
                a01 += wa * xb;
                a11 += wb * xb;
            }
        }

        acc[0] = a00;
        acc[1] = a10;
        acc[2] = a01;
        acc[3] = a11;
    }

} } } // namespace tinymind::simd::arm_dsp

#endif // __ARM_FEATURE_DSP

#endif // TINYMIND_ENABLE_SIMD_ARM_DSP
