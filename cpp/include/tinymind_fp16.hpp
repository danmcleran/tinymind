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

#include "tinymind_platform.hpp"

#include <cstddef>
#include <cstdint>

/*
 * Phase 9 half-precision storage tier.
 *
 * Two software storage types backed by a uint16_t:
 *
 *   fp16_t  IEEE 754 binary16  (1 sign, 5 exponent, 10 mantissa)
 *   bf16_t  bfloat16            (1 sign, 8 exponent, 7 mantissa, upper
 *                                 16 bits of fp32)
 *
 * Conversion to/from float is scalar and freestanding-safe (no <cmath>,
 * no <cstring>, no namespace std). Bit-level punning uses
 * __builtin_memcpy, available on gcc / clang, which the rest of the
 * library already targets.
 *
 * Storage only. SIMD specializations (NEON fp16, AVX-512 fp16, x86 fp16C
 * intrinsics) land in Phase 14 behind separate gates. Hosts that
 * natively support _Float16 / __fp16 may add a thin adapter without
 * disturbing this header.
 *
 * Gated on TINYMIND_ENABLE_FP16. Conversion helpers additionally require
 * TINYMIND_ENABLE_FLOAT because they round through float; the storage
 * structs themselves compile unconditionally when the file is included.
 */

#if TINYMIND_ENABLE_FP16

namespace tinymind {

    struct fp16_t
    {
        uint16_t bits;
    };

    struct bf16_t
    {
        uint16_t bits;
    };

#if TINYMIND_ENABLE_FLOAT

    /**
     * bfloat16 <-> float.
     *
     * bf16 is the upper 16 bits of fp32. float -> bf16 rounds the
     * truncated bits to nearest, ties to even. bf16 -> float is a pure
     * left shift; no rounding loss.
     */
    inline bf16_t floatToBf16(float x)
    {
        uint32_t bits;
        __builtin_memcpy(&bits, &x, 4);

        // NaN preservation: any NaN encodes to a quiet NaN with the high
        // mantissa bit set.
        const uint32_t exp_field = (bits >> 23) & 0xFFu;
        const uint32_t mantissa  = bits & 0x7FFFFFu;
        if (exp_field == 0xFFu && mantissa != 0)
        {
            const uint16_t nan_bits = static_cast<uint16_t>(
                ((bits >> 16) & 0xFFFFu) | static_cast<uint16_t>(0x0040u));
            bf16_t out;
            out.bits = nan_bits;
            return out;
        }

        // Round to nearest, ties to even.
        const uint32_t lsb = (bits >> 16) & 1u;
        const uint32_t rounding_bias = 0x7FFFu + lsb;
        const uint32_t rounded = bits + rounding_bias;

        bf16_t out;
        out.bits = static_cast<uint16_t>(rounded >> 16);
        return out;
    }

    inline float bf16ToFloat(bf16_t x)
    {
        uint32_t bits = static_cast<uint32_t>(x.bits) << 16;
        float out;
        __builtin_memcpy(&out, &bits, 4);
        return out;
    }

    /**
     * IEEE 754 binary16 <-> float.
     *
     * Handles zero, subnormal, normal, Inf, and NaN. Rounds to nearest,
     * ties to even on the truncated mantissa bits. Overflow snaps to
     * signed Inf; underflow saturates toward zero with subnormal
     * handling.
     */
    inline fp16_t floatToFp16(float x)
    {
        uint32_t f;
        __builtin_memcpy(&f, &x, 4);

        const uint32_t sign     = (f >> 31) & 0x1u;
        const uint32_t exp_f    = (f >> 23) & 0xFFu;
        const uint32_t mant_f   = f & 0x7FFFFFu;

        const uint16_t h_sign = static_cast<uint16_t>(sign << 15);

        if (exp_f == 0xFFu)
        {
            // Inf or NaN.
            const uint16_t h_mant = static_cast<uint16_t>(mant_f ? 0x0200u : 0u);
            fp16_t out;
            out.bits = static_cast<uint16_t>(h_sign | 0x7C00u | h_mant);
            return out;
        }

        const int32_t new_exp = static_cast<int32_t>(exp_f) - 127 + 15;

        if (new_exp >= 0x1F)
        {
            // Overflow -> Inf.
            fp16_t out;
            out.bits = static_cast<uint16_t>(h_sign | 0x7C00u);
            return out;
        }

        if (new_exp <= 0)
        {
            // Subnormal or zero.
            if (new_exp < -10)
            {
                fp16_t out;
                out.bits = h_sign;
                return out;
            }
            // Add the implicit leading 1 and shift to fp16 subnormal slot.
            uint32_t mant_implicit = mant_f | 0x800000u;
            const uint32_t shift = static_cast<uint32_t>(14 - new_exp);
            uint16_t h_mant = static_cast<uint16_t>(mant_implicit >> shift);

            // Round to nearest, ties to even.
            const uint32_t rem  = mant_implicit & ((1u << shift) - 1u);
            const uint32_t half = 1u << (shift - 1u);
            if (rem > half || (rem == half && (h_mant & 1u)))
            {
                ++h_mant;
            }

            fp16_t out;
            out.bits = static_cast<uint16_t>(h_sign | h_mant);
            return out;
        }

        uint16_t h_exp  = static_cast<uint16_t>(new_exp << 10);
        uint16_t h_mant = static_cast<uint16_t>(mant_f >> 13);

        // Round to nearest, ties to even on the dropped 13 bits.
        const uint32_t rem = mant_f & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (h_mant & 1u)))
        {
            ++h_mant;
            if (h_mant & 0x0400u)
            {
                // Mantissa overflowed; bump the exponent.
                h_mant = 0;
                h_exp = static_cast<uint16_t>(h_exp + (1u << 10));
                if (h_exp >= 0x7C00u)
                {
                    fp16_t out;
                    out.bits = static_cast<uint16_t>(h_sign | 0x7C00u);
                    return out;
                }
            }
        }

        fp16_t out;
        out.bits = static_cast<uint16_t>(h_sign | h_exp | h_mant);
        return out;
    }

    inline float fp16ToFloat(fp16_t x)
    {
        const uint32_t h        = static_cast<uint32_t>(x.bits);
        const uint32_t sign     = (h >> 15) & 0x1u;
        const uint32_t exp_h    = (h >> 10) & 0x1Fu;
        const uint32_t mant_h   = h & 0x3FFu;

        const uint32_t f_sign = sign << 31;
        uint32_t f;

        if (exp_h == 0u)
        {
            if (mant_h == 0u)
            {
                f = f_sign;
            }
            else
            {
                // Subnormal: renormalize.
                uint32_t mant = mant_h;
                int32_t exp = 1;
                while ((mant & 0x400u) == 0u)
                {
                    mant <<= 1;
                    --exp;
                }
                mant &= 0x3FFu;
                const uint32_t f_exp = static_cast<uint32_t>(exp + 127 - 15) << 23;
                f = f_sign | f_exp | (mant << 13);
            }
        }
        else if (exp_h == 0x1Fu)
        {
            // Inf or NaN.
            f = f_sign | 0x7F800000u | (mant_h << 13);
        }
        else
        {
            const uint32_t f_exp = (exp_h + 127u - 15u) << 23;
            f = f_sign | f_exp | (mant_h << 13);
        }

        float out;
        __builtin_memcpy(&out, &f, 4);
        return out;
    }

    /**
     * Buffer-batch conversions. Caller-owned storage; no allocation.
     */
    inline void floatToBf16Buffer(const float* src, bf16_t* dst, std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            dst[i] = floatToBf16(src[i]);
        }
    }

    inline void bf16ToFloatBuffer(const bf16_t* src, float* dst, std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            dst[i] = bf16ToFloat(src[i]);
        }
    }

    inline void floatToFp16Buffer(const float* src, fp16_t* dst, std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            dst[i] = floatToFp16(src[i]);
        }
    }

    inline void fp16ToFloatBuffer(const fp16_t* src, float* dst, std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            dst[i] = fp16ToFloat(src[i]);
        }
    }

#endif // TINYMIND_ENABLE_FLOAT

} // namespace tinymind

#endif // TINYMIND_ENABLE_FP16
