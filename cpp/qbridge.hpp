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
 * Two flavors are exposed:
 *
 *   1. Float-mediated bridges (this file, gated on TINYMIND_ENABLE_FLOAT).
 *      Conversions pass through a float intermediate, so the per-sample
 *      math matches the calibration-time math byte-for-byte. Available on
 *      any target with an FPU.
 *
 *   2. Pure-integer bridges (this file, gated on TINYMIND_ENABLE_QUANTIZATION).
 *      Conversions use the same (multiplier, shift) Q0.31 primitive that
 *      Requantizer uses. No <cmath>, no <type_traits>, no float at runtime.
 *      The deployable freestanding shape FLOAT=0 STD=0 QUANT=1 picks this
 *      path. Caller builds the conversion params host-side once via the
 *      buildAffineToQValueIntParams / buildQValueToAffineIntParams helpers
 *      (FLOAT && STD gated) and ships the resulting integer triple to the
 *      target along with the rest of the quantized model.
 *
 * Bridges intentionally do NOT touch layer internals. They are pointwise
 * value converters; cascading them through a buffer is the caller's job.
 */

#if TINYMIND_ENABLE_QUANTIZATION

#include "qaffine.hpp"
#include "qformat.hpp"

namespace tinymind {

    /**
     * Pure-integer affine -> Q-format conversion params.
     *
     * qval = round(scale * (q - zero_point) * 2^F) saturated to QValue range,
     * with the scale * 2^F factor encoded as a (multiplier, shift) Q0.31 pair.
     * Built host-side by buildAffineToQValueIntParams (FLOAT && STD); shipped
     * to the target as three int32s.
     */
    template<typename QV>
    struct AffineToQValueIntParams
    {
        int32_t multiplier;
        int32_t shift;
        int32_t zero_point;
    };

    /**
     * Pure-integer Q-format -> affine conversion params.
     *
     * q = round(qval / (scale * 2^F)) + zero_point clamped to [qmin, qmax].
     * The reciprocal-scale factor 1 / (scale * 2^F) is encoded as a
     * (multiplier, shift) pair.
     */
    template<typename QV>
    struct QValueToAffineIntParams
    {
        int32_t multiplier;
        int32_t shift;
        int32_t zero_point;
        int32_t qmin;
        int32_t qmax;
    };

    namespace detail {

        /**
         * Saturating cast of a Q-format raw value to int32. QValue raw types
         * up to int32 pass through unchanged; the wider Q32.32 / Q64.0 raw
         * type saturates here rather than wrapping. Callers needing the full
         * dynamic range of a wide Q-format on the target should pre-narrow
         * before the bridge.
         */
        inline int32_t qBridgeRawToInt32Sat(int64_t raw)
        {
            const int64_t hi = static_cast<int64_t>(0x7FFFFFFF);
            const int64_t lo = -hi - 1;
            if (raw > hi) return static_cast<int32_t>(hi);
            if (raw < lo) return static_cast<int32_t>(lo);
            return static_cast<int32_t>(raw);
        }

    } // namespace detail

    /**
     * affine int8 -> Q-format, pure-integer path. Inverse of
     * qValueToAffineInt. Composes multiplyByQuantizedMultiplier on the
     * zero-point-removed accumulator and saturates to the QValue range.
     */
    template<typename QV, typename SrcStorage>
    inline QV affineToQValueInt(SrcStorage q, const AffineToQValueIntParams<QV>& p)
    {
        typedef typename QV::FullWidthValueType RawType;
        const int32_t shifted = static_cast<int32_t>(q) - p.zero_point;
        const int32_t scaled = multiplyByQuantizedMultiplier(shifted, p.multiplier, p.shift);

        const int64_t lo = static_cast<int64_t>(QV::QFormatMinValue());
        const int64_t hi = static_cast<int64_t>(QV::QFormatMaxValue());
        int64_t out = static_cast<int64_t>(scaled);
        if (out < lo) out = lo;
        if (out > hi) out = hi;

        return QV(static_cast<RawType>(out));
    }

    /**
     * Q-format -> affine int8, pure-integer path. Multiplies the raw
     * QValue by the precomputed reciprocal-scale (multiplier, shift),
     * adds the destination zero_point, and saturates to [qmin, qmax].
     */
    template<typename QV, typename DstStorage>
    inline DstStorage qValueToAffineInt(const QV& qv, const QValueToAffineIntParams<QV>& p)
    {
        const int32_t raw = detail::qBridgeRawToInt32Sat(
            static_cast<int64_t>(qv.getValue()));
        const int32_t scaled = multiplyByQuantizedMultiplier(raw, p.multiplier, p.shift);
        int32_t with_zp = scaled + p.zero_point;
        if (with_zp < p.qmin) with_zp = p.qmin;
        if (with_zp > p.qmax) with_zp = p.qmax;
        return static_cast<DstStorage>(with_zp);
    }

    template<typename QV, typename SrcStorage>
    inline void affineToQValueIntBuffer(const SrcStorage* src, QV* dst,
                                        std::size_t n,
                                        const AffineToQValueIntParams<QV>& p)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            dst[i] = affineToQValueInt<QV, SrcStorage>(src[i], p);
        }
    }

    template<typename QV, typename DstStorage>
    inline void qValueToAffineIntBuffer(const QV* src, DstStorage* dst,
                                        std::size_t n,
                                        const QValueToAffineIntParams<QV>& p)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            dst[i] = qValueToAffineInt<QV, DstStorage>(src[i], p);
        }
    }

} // namespace tinymind

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD

#include <cmath>

namespace tinymind {

    /**
     * Build the integer affine -> Q-format conversion params from the
     * affine (scale, zero_point) calibration metadata. Host-only; the
     * resulting struct is what the target consumes.
     */
    template<typename QV>
    inline AffineToQValueIntParams<QV>
    buildAffineToQValueIntParams(float scale, int32_t zero_point)
    {
        const double frac =
            static_cast<double>(static_cast<uint64_t>(1) << QV::NumberOfFractionalBits);
        const double ratio = static_cast<double>(scale) * frac;
        AffineToQValueIntParams<QV> p;
        quantizeMultiplier(ratio, p.multiplier, p.shift);
        p.zero_point = zero_point;
        return p;
    }

    /**
     * Build the integer Q-format -> affine conversion params from the
     * downstream affine (scale, zero_point) plus the destination saturation
     * range. The encoded ratio is 1 / (scale * 2^F); the destination
     * zero_point and qmin/qmax pass through unchanged.
     */
    template<typename QV>
    inline QValueToAffineIntParams<QV>
    buildQValueToAffineIntParams(float scale, int32_t zero_point,
                                 int32_t qmin, int32_t qmax)
    {
        const double frac =
            static_cast<double>(static_cast<uint64_t>(1) << QV::NumberOfFractionalBits);
        const double ratio = 1.0 / (static_cast<double>(scale) * frac);
        QValueToAffineIntParams<QV> p;
        quantizeMultiplier(ratio, p.multiplier, p.shift);
        p.zero_point = zero_point;
        p.qmin = qmin;
        p.qmax = qmax;
        return p;
    }

} // namespace tinymind

#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD

#endif // TINYMIND_ENABLE_QUANTIZATION


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
