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
#include "include/simd/simd_dispatch.hpp"
#include "qaffine.hpp"

#include <cstddef>
#include <cstdint>

/*
 * Quantized fully-connected (Dense) layer.
 *
 * Standalone layer that mirrors the affine quantization model used by
 * TFLite reference / CMSIS-NN kernels. It does NOT integrate with the
 * NeuralNet<> template; weights, biases, and the per-output Requantizer
 * are caller-owned and the forward pass is a single function call.
 *
 * Type contract (matches the canonical int8 deployment shape):
 *
 *   InputStorage  = int8_t   asymmetric activations (zero_point baked into
 *                            input_zero_point at calibration)
 *   WeightStorage = int8_t   symmetric weights (zero_point = 0)
 *   AccumType     = int32_t  bias and accumulator
 *   OutputStorage = int8_t   asymmetric output (handled inside Requantizer)
 *
 * The forward pass evaluates:
 *
 *   acc[o] = bias[o] + sum_i weight[o,i] * (input[i] - input_zero_point)
 *   output[o] = Requantizer.apply(acc[o])
 *
 * Bias is held in the accumulator type with effective scale equal to
 * input_scale * weight_scale, so it is added directly to the partial sum
 * without further rescaling. The Requantizer applies the
 *   effective_ratio = (input_scale * weight_scale) / output_scale
 * decomposition computed at calibration time.
 *
 * weights is a row-major buffer of shape [NumOutputs * NumInputs].
 * biases may be nullptr to indicate "no bias"; in that case the
 * accumulator starts at zero.
 *
 * Pure integer at runtime; no float, no <cmath>, no stdlib. Safe to
 * compile in the freestanding (FLOAT=0, STD=0) configuration.
 */

namespace tinymind {

#if defined(TINYMIND_SIMD_INT8_DOT_ROWS4)
    namespace detail {

        template<bool Blocked>
        struct Rows4Tag
        {
        };

        /**
         * A dense layer (or a single-pixel pointwise layer) four outputs at a
         * time through simd::DotRows4, which widens each input word once for
         * the four weight rows; the last NumOutputs % 4 outputs take one dot
         * product each. rqs[o * RqStride] is output o's requantizer (stride 0
         * for one per-tensor requantizer, 1 for one per output). The same
         * sums and the same requantizer per output as the one-row loop, so
         * bit-exact with it. Only compiled when the Arm DSP backend is
         * dispatched; Rows4Tag<false> keeps the caller's own loop.
         */
        template<typename Input, typename Weight, typename Accum, typename Output,
                 std::size_t NumInputs, std::size_t NumOutputs, std::size_t RqStride>
        bool denseRows4(const Input*, Output*, const Weight*, const Accum*, Input,
                        const Requantizer<Accum, Output>*, Rows4Tag<false>)
        {
            return false;
        }

        template<typename Input, typename Weight, typename Accum, typename Output,
                 std::size_t NumInputs, std::size_t NumOutputs, std::size_t RqStride>
        bool denseRows4(const Input* input, Output* output, const Weight* weights, const Accum* biases,
                        Input zp, const Requantizer<Accum, Output>* rqs, Rows4Tag<true>)
        {
            constexpr std::size_t Blocked = NumOutputs - NumOutputs % 4;
            for (std::size_t o = 0; o < Blocked; o += 4)
            {
                Accum acc[4];
                for (std::size_t k = 0; k < 4; ++k)
                {
                    acc[k] = (biases != nullptr) ? biases[o + k] : static_cast<Accum>(0);
                }
                simd::DotRows4<Input, Weight, Accum, NumInputs, NumInputs>::apply(
                    input, weights + o * NumInputs, zp, acc);
                for (std::size_t k = 0; k < 4; ++k)
                {
                    output[o + k] = rqs[(o + k) * RqStride].apply(acc[k]);
                }
            }
            for (std::size_t o = Blocked; o < NumOutputs; ++o)
            {
                const Accum bias = (biases != nullptr) ? biases[o] : static_cast<Accum>(0);
                const Accum acc = bias + simd::dotProductWithZeroPoint<Input, Weight, Accum>(
                    input, weights + o * NumInputs, NumInputs, zp);
                output[o] = rqs[o * RqStride].apply(acc);
            }
            return true;
        }

        template<typename Input, typename Weight, typename Accum, std::size_t NumInputs, std::size_t NumOutputs>
        struct UseRows4
        {
            static constexpr bool value =
                simd::DotRows4<Input, Weight, Accum, NumInputs, NumInputs>::kBlocked && (NumOutputs >= 4);
        };

    } // namespace detail
#endif

    template<typename InputStorage_, typename WeightStorage_, typename AccumType_,
             typename OutputStorage_, std::size_t NumInputs_, std::size_t NumOutputs_>
    struct QDense
    {
        typedef InputStorage_ InputType;
        typedef WeightStorage_ WeightType;
        typedef AccumType_ AccumulatorType;
        typedef OutputStorage_ OutputType;

        static constexpr std::size_t InputLength = NumInputs_;
        static constexpr std::size_t OutputLength = NumOutputs_;

        const WeightType* weights;
        const AccumulatorType* biases;
        InputType input_zero_point;
        Requantizer<AccumulatorType, OutputType> requantizer;

        void forward(const InputType* input, OutputType* output) const
        {
            // Snapshot the fields into locals. Every output store is an int8_t write,
            // which may alias *this, so member reads inside the loops would be
            // reloaded -- and the requantizer's shift re-decoded -- once per output.
            const WeightType* const layerWeights = weights;
            const AccumulatorType* const layerBiases = biases;
            const InputType inputZeroPoint = input_zero_point;
            const Requantizer<AccumulatorType, OutputType> rq = requantizer;

#if defined(TINYMIND_SIMD_INT8_DOT_ROWS4)
            if (detail::denseRows4<InputType, WeightType, AccumulatorType, OutputType, NumInputs_, NumOutputs_, 0>(
                    input, output, layerWeights, layerBiases, inputZeroPoint, &rq,
                    detail::Rows4Tag<detail::UseRows4<InputType, WeightType, AccumulatorType,
                                                      NumInputs_, NumOutputs_>::value>()))
            {
                return;
            }
#endif

#if TINYMIND_SPLIT_REQUANTIZER_FORMS
            // The same loop without the saturating path's code (see
            // TINYMIND_SPLIT_REQUANTIZER_FORMS in qaffine.hpp).
            if (rq.oneRounding())
            {
                for (std::size_t o = 0; o < NumOutputs_; ++o)
                {
                    AccumulatorType acc = (layerBiases != nullptr)
                        ? layerBiases[o]
                        : static_cast<AccumulatorType>(0);

                    const WeightType* w_row = layerWeights + o * NumInputs_;
                    acc += simd::dotProductWithZeroPoint<InputType, WeightType,
                                                         AccumulatorType>(
                        input, w_row, NumInputs_, inputZeroPoint);

                    output[o] = rq.applyOneRounding(acc);
                }
                return;
            }
#endif

            for (std::size_t o = 0; o < NumOutputs_; ++o)
            {
                AccumulatorType acc = (layerBiases != nullptr)
                    ? layerBiases[o]
                    : static_cast<AccumulatorType>(0);

                const WeightType* w_row = layerWeights + o * NumInputs_;
                acc += simd::dotProductWithZeroPoint<InputType, WeightType,
                                                     AccumulatorType>(
                    input, w_row, NumInputs_, inputZeroPoint);

                output[o] = rq.apply(acc);
            }
        }
    };

} // namespace tinymind
