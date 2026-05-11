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

#include "include/tinymind_platform.hpp"

#include <cstddef>
#include <cstdint>

/*
 * Phase 9 type bridges.
 *
 * Q-format (QValue) and affine int8 (QAffineTensor) coexisted as
 * independent pipelines before this file. qbridge.hpp adds the missing
 * boundary converters so a model can mix:
 *
 *   int8 affine CNN frontend  ->  Q-format LSTM head  ->  int8 affine classifier
 *   float input               ->  int8 affine layer
 *   int8 affine output        ->  fp16 / bf16 head
 *
 * All conversions run scalar at the layer boundary; the inner loops of
 * each pipeline stay native to their own type system. The conversions
 * themselves use float arithmetic, so this whole file is gated on
 * TINYMIND_ENABLE_FLOAT. Any target whose silicon ships an FPU (capability,
 * not CPU model — Arm publishes thousands of RTL configurations per core
 * and FPU is often an optional component) can use the bridges at runtime;
 * a pure-integer build keeps both pipelines siloed by simply not
 * including this header.
 *
 * No <cmath> dependency: rounding uses sign-aware float-to-int casting.
 * No <type_traits>. Freestanding-safe at FLOAT=1, STD=0.
 *
 * Bridges intentionally do NOT touch layer internals. They are pointwise
 * value converters; cascading them through a buffer is the caller's job.
 */

#if TINYMIND_ENABLE_FLOAT

#include "qaffine.hpp"

#if TINYMIND_ENABLE_FP16
#include "include/tinymind_fp16.hpp"
#endif

namespace tinymind {

    namespace detail {

        /**
         * Round-half-away-from-zero, freestanding. Avoids std::lround so
         * the bridge stays usable at STD=0.
         */
        inline int32_t roundFloatToInt(float x)
        {
            return (x >= 0.0f)
                ? static_cast<int32_t>(x + 0.5f)
                : static_cast<int32_t>(x - 0.5f);
        }

        template<typename FloatType>
        inline FloatType pow2NegativeFloatBridge(unsigned exponent)
        {
            FloatType result = static_cast<FloatType>(1);
            for (unsigned i = 0; i < exponent; ++i)
            {
                result *= static_cast<FloatType>(0.5);
            }
            return result;
        }

    } // namespace detail

    /**
     * Dequantize a single affine sample to float.
     *
     * real = scale * (q - zero_point). Caller supplies the affine
     * parameters that were emitted by host calibration; the helper is
     * format-agnostic in the storage type.
     */
    template<typename SrcStorage>
    inline float affineDequantize(SrcStorage q, float scale, int32_t zero_point)
    {
        const int32_t shifted = static_cast<int32_t>(q) - zero_point;
        return scale * static_cast<float>(shifted);
    }

    /**
     * Quantize a float to the destination affine storage type with
     * saturation. Rounding follows round-half-away-from-zero (matches
     * the calibration helpers in qcalibration.hpp).
     */
    template<typename DstStorage>
    inline DstStorage affineQuantize(float x, float scale, int32_t zero_point,
                                     int32_t qmin, int32_t qmax)
    {
        const float reciprocal_scale = 1.0f / scale;
        int32_t q = detail::roundFloatToInt(x * reciprocal_scale) + zero_point;
        if (q < qmin) q = qmin;
        if (q > qmax) q = qmax;
        return static_cast<DstStorage>(q);
    }

    /**
     * Q-format -> float. QValue uses a compile-time fractional bit count
     * so the scale factor is a fixed power of two; no calibration
     * metadata required.
     */
    template<typename QV>
    inline float qValueToFloat(const QV& qv)
    {
        const float factor =
            detail::pow2NegativeFloatBridge<float>(QV::NumberOfFractionalBits);
        return static_cast<float>(qv.getValue()) * factor;
    }

    /**
     * float -> Q-format. Raw value = round(x * 2^frac) saturated to the
     * Q-format full-width range.
     */
    template<typename QV>
    inline QV floatToQValue(float x)
    {
        typedef typename QV::FullWidthValueType RawType;
        const float scale = static_cast<float>(
            static_cast<uint64_t>(1) << QV::NumberOfFractionalBits);
        int64_t raw = static_cast<int64_t>(detail::roundFloatToInt(x * scale));

        const int64_t lo = static_cast<int64_t>(QV::QFormatMinValue());
        const int64_t hi = static_cast<int64_t>(QV::QFormatMaxValue());
        if (raw < lo) raw = lo;
        if (raw > hi) raw = hi;

        return QV(static_cast<RawType>(raw));
    }

    /**
     * Q-format -> int8 affine. Bridges the existing fixed-point pipeline
     * into a downstream affine layer at a layer boundary. Composes the
     * Q->float and float->affine helpers; no extra precision loss
     * beyond the one rounding step.
     */
    template<typename QV, typename DstStorage>
    inline DstStorage qValueToAffine(const QV& qv, float scale, int32_t zero_point,
                                     int32_t qmin, int32_t qmax)
    {
        const float as_float = qValueToFloat<QV>(qv);
        return affineQuantize<DstStorage>(as_float, scale, zero_point, qmin, qmax);
    }

    /**
     * int8 affine -> Q-format. Inverse of qValueToAffine. Useful for
     * feeding int8 conv output into a Q-format recurrent or attention
     * head.
     */
    template<typename QV, typename SrcStorage>
    inline QV affineToQValue(SrcStorage q, float scale, int32_t zero_point)
    {
        const float as_float = affineDequantize<SrcStorage>(q, scale, zero_point);
        return floatToQValue<QV>(as_float);
    }

    /**
     * Buffer-batch variants. Caller-owned destination storage.
     */
    template<typename SrcStorage>
    inline void affineDequantizeBuffer(const SrcStorage* src, float* dst,
                                       std::size_t n, float scale, int32_t zero_point)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            dst[i] = affineDequantize<SrcStorage>(src[i], scale, zero_point);
        }
    }

    template<typename DstStorage>
    inline void affineQuantizeBuffer(const float* src, DstStorage* dst,
                                     std::size_t n, float scale, int32_t zero_point,
                                     int32_t qmin, int32_t qmax)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            dst[i] = affineQuantize<DstStorage>(src[i], scale, zero_point, qmin, qmax);
        }
    }

    template<typename QV>
    inline void qValueToFloatBuffer(const QV* src, float* dst, std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            dst[i] = qValueToFloat<QV>(src[i]);
        }
    }

    template<typename QV>
    inline void floatToQValueBuffer(const float* src, QV* dst, std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            dst[i] = floatToQValue<QV>(src[i]);
        }
    }

    template<typename QV, typename DstStorage>
    inline void qValueToAffineBuffer(const QV* src, DstStorage* dst, std::size_t n,
                                     float scale, int32_t zero_point,
                                     int32_t qmin, int32_t qmax)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            dst[i] = qValueToAffine<QV, DstStorage>(src[i], scale, zero_point,
                                                    qmin, qmax);
        }
    }

    template<typename QV, typename SrcStorage>
    inline void affineToQValueBuffer(const SrcStorage* src, QV* dst, std::size_t n,
                                     float scale, int32_t zero_point)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            dst[i] = affineToQValue<QV, SrcStorage>(src[i], scale, zero_point);
        }
    }

#if TINYMIND_ENABLE_FP16

    /**
     * fp16 / bf16 <-> int8 affine bridges.
     *
     * Promote half-precision input to float, then quantize; dequantize
     * to float, then narrow. The float intermediate is intentional: it
     * keeps the affine rounding identical to the float path so a model
     * with mixed fp16 / int8 layers behaves the same as one trained
     * entirely in fp32.
     */
    inline int8_t fp16ToAffineI8(fp16_t x, float scale, int32_t zero_point,
                                 int32_t qmin, int32_t qmax)
    {
        return affineQuantize<int8_t>(fp16ToFloat(x), scale, zero_point, qmin, qmax);
    }

    inline fp16_t affineI8ToFp16(int8_t q, float scale, int32_t zero_point)
    {
        return floatToFp16(affineDequantize<int8_t>(q, scale, zero_point));
    }

    inline int8_t bf16ToAffineI8(bf16_t x, float scale, int32_t zero_point,
                                 int32_t qmin, int32_t qmax)
    {
        return affineQuantize<int8_t>(bf16ToFloat(x), scale, zero_point, qmin, qmax);
    }

    inline bf16_t affineI8ToBf16(int8_t q, float scale, int32_t zero_point)
    {
        return floatToBf16(affineDequantize<int8_t>(q, scale, zero_point));
    }

#endif // TINYMIND_ENABLE_FP16

} // namespace tinymind

#endif // TINYMIND_ENABLE_FLOAT
