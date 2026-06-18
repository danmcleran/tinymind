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
#include "qdense.hpp"
#include "qsoftmax.hpp"

#include <cstddef>
#include <cstdint>

/*
 * Quantized Mixture-of-Experts layer with top-1 (Switch-style) routing.
 *
 * Standalone int8 inference layer following the same caller-owned, aggregate
 * model as QDense -- weights, biases and per-expert Requantizers live with the
 * caller; the forward pass is a single function call. Pure integer at runtime;
 * no float, no <cmath>, no stdlib. Safe in the freestanding (FLOAT=0, STD=0)
 * configuration.
 *
 * Two parts:
 *
 *   Router  -- a linear layer NumInputs -> NumExperts. It accumulates int32
 *              logits exactly like QDense but does NOT requantize: the expert
 *              is chosen by argmax over the raw int32 logits. argmax over the
 *              logits is identical to argmax over their softmax (softmax is
 *              monotonic), so no softmax is needed, and the choice is robust to
 *              quantization noise -- it only flips on a near-tie, where both
 *              experts produce similar outputs anyway. Keeping the router in
 *              int32 with no requant avoids injecting requant error into the
 *              decision.
 *
 *   Experts -- NumExperts independently-calibrated QDense layers. Each carries
 *              its own weights, bias and Requantizer, i.e. its own per-expert
 *              quantization scale. All experts are resident; exactly one runs
 *              per call (the argmax winner). Active compute equals one expert;
 *              storage equals the sum of all experts plus the router.
 *
 * The forward pass evaluates:
 *
 *   logit[e] = router_bias[e] + sum_i router_weight[e,i] * (input[i] - input_zero_point)
 *   e*       = argmax_e logit[e]        (ties resolve to the lowest index)
 *   output   = experts[e*].forward(input)
 *
 * No gate scaling is applied: fold any gate weight into the expert during
 * training, or apply it externally using the logits from route(). This phase
 * is inference-and-weight-loading oriented; training (which must contend with
 * the non-differentiable argmax and load balancing) is a host/PyTorch concern.
 *
 * router_weights is a row-major buffer of shape [NumExperts * NumInputs].
 * router_biases may be nullptr to indicate "no bias"; logits then start at zero.
 */

namespace tinymind {

    template<typename InputStorage_, typename WeightStorage_, typename AccumType_,
             typename OutputStorage_, std::size_t NumInputs_, std::size_t NumOutputs_,
             std::size_t NumExperts_>
    struct QMixtureOfExperts
    {
        typedef InputStorage_ InputType;
        typedef WeightStorage_ WeightType;
        typedef AccumType_ AccumulatorType;
        typedef OutputStorage_ OutputType;
        typedef QDense<InputStorage_, WeightStorage_, AccumType_, OutputStorage_,
                       NumInputs_, NumOutputs_> Expert;

        static constexpr std::size_t InputLength = NumInputs_;
        static constexpr std::size_t OutputLength = NumOutputs_;
        static constexpr std::size_t NumberOfExperts = NumExperts_;

        // Router (no requantizer; argmax over raw int32 logits)
        const WeightType* router_weights;     // [NumExperts * NumInputs], row-major
        const AccumulatorType* router_biases; // [NumExperts] or nullptr
        InputType input_zero_point;

        // Experts: each a fully-configured QDense (per-expert scale)
        Expert experts[NumExperts_];

        /**
         * Compute router logits and select the top-1 expert without running it.
         * @param input  NumInputs int8 activations
         * @param logits NumExperts int32 logits, filled if non-null
         * @return index of the argmax (selected) expert
         */
        std::size_t route(const InputType* input, AccumulatorType* logits) const
        {
            std::size_t best = 0;
            AccumulatorType bestLogit = static_cast<AccumulatorType>(0);

            for (std::size_t e = 0; e < NumExperts_; ++e)
            {
                AccumulatorType acc = (router_biases != nullptr)
                    ? router_biases[e]
                    : static_cast<AccumulatorType>(0);

                const WeightType* w_row = router_weights + e * NumInputs_;
                acc += simd::dotProductWithZeroPoint<InputType, WeightType,
                                                     AccumulatorType>(
                    input, w_row, NumInputs_, input_zero_point);

                if (logits != nullptr)
                {
                    logits[e] = acc;
                }

                if (e == 0 || acc > bestLogit)
                {
                    bestLogit = acc;
                    best = e;
                }
            }

            return best;
        }

        /**
         * Top-1 forward: route to the argmax expert and run it.
         * @param input  NumInputs int8 activations
         * @param output NumOutputs int8 values, filled by the selected expert
         * @return index of the selected expert
         */
        std::size_t forward(const InputType* input, OutputType* output) const
        {
            const std::size_t best = route(input, nullptr);
            experts[best].forward(input, output);
            return best;
        }

        static_assert(NumExperts_ > 0, "Number of experts must be > 0.");
        static_assert(NumInputs_ > 0, "Number of inputs must be > 0.");
        static_assert(NumOutputs_ > 0, "Number of outputs must be > 0.");
    };

    /**
     * Quantized top-k / dense Mixture-of-Experts with fixed-point gated blend.
     *
     * Runs the K highest-scoring experts and returns their gate-weighted sum,
     * the pure-integer counterpart of moe.hpp::TopKMixtureOfExperts:
     *
     *   topk    = indices of the K largest int32 router logits
     *   g_i     = softmax over the requantized top-k logits (via exp LUT)
     *   output  = sum_{i in topk} g_i * expert_i(x)
     *
     * K == 1           collapses to top-1 (gate 1.0; same result as
     *                  QMixtureOfExperts::forward).
     * K == NumExperts  is a dense MoE.
     *
     * Two precisions of router logit are used, deliberately:
     *   * top-k SELECTION uses the raw int32 logits (scale-invariant argmax /
     *     partial sort -- robust to quantization noise, no requant needed);
     *   * the GATE WEIGHTS need real softmax, so the selected logits are
     *     requantized to int8 (router_requantizer) and fed through exp_lut,
     *     a 256-entry table built by qcalibration.hpp::buildQSoftmaxExpLUT at
     *     the router logit scale. The same qSoftmaxLUTIndex convention as
     *     QSoftmax1D keeps the table reusable.
     *
     * Experts share one output scale / zero point (output_zero_point), so
     * their int8 outputs live in one affine grid and can be blended directly:
     * the gate-weighted sum is accumulated over (y_i - output_zero_point) in
     * Q15, shifted back, re-centred on output_zero_point, and saturated to
     * [output_qmin, output_qmax]. Pure integer; freestanding-safe.
     *
     * forward() returns the top-1 (highest-logit) expert index.
     */
    template<typename InputStorage_, typename WeightStorage_, typename AccumType_,
             typename OutputStorage_, std::size_t NumInputs_, std::size_t NumOutputs_,
             std::size_t NumExperts_, std::size_t K_>
    struct QTopKMixtureOfExperts
    {
        typedef InputStorage_ InputType;
        typedef WeightStorage_ WeightType;
        typedef AccumType_ AccumulatorType;
        typedef OutputStorage_ OutputType;
        typedef QDense<InputStorage_, WeightStorage_, AccumType_, OutputStorage_,
                       NumInputs_, NumOutputs_> Expert;

        static constexpr std::size_t InputLength = NumInputs_;
        static constexpr std::size_t OutputLength = NumOutputs_;
        static constexpr std::size_t NumberOfExperts = NumExperts_;
        static constexpr std::size_t TopK = K_;

        // Gate weights are accumulated in Q(kGateShift).
        static constexpr int kGateShift = 15;

        // Router: int32 logits (raw, for selection) + a requantizer that maps
        // a selected logit to int8 for the exp lookup.
        const WeightType* router_weights;       // [NumExperts * NumInputs]
        const AccumulatorType* router_biases;   // [NumExperts] or nullptr
        InputType input_zero_point;
        Requantizer<AccumulatorType, OutputType> router_requantizer;
        const int32_t* exp_lut;                 // [256], buildQSoftmaxExpLUT

        // Experts: each its own weight scale, all sharing the output scale.
        Expert experts[NumExperts_];

        // Shared output affine grid for the blend.
        OutputType output_zero_point;
        OutputType output_qmin;
        OutputType output_qmax;

        std::size_t forward(const InputType* input, OutputType* output) const
        {
            // 1. Raw int32 router logits.
            AccumulatorType logits[NumExperts_];
            for (std::size_t e = 0; e < NumExperts_; ++e)
            {
                AccumulatorType acc = (router_biases != nullptr)
                    ? router_biases[e]
                    : static_cast<AccumulatorType>(0);
                const WeightType* w_row = router_weights + e * NumInputs_;
                acc += simd::dotProductWithZeroPoint<InputType, WeightType,
                                                     AccumulatorType>(
                    input, w_row, NumInputs_, input_zero_point);
                logits[e] = acc;
            }

            std::size_t idx[K_];
            selectTopK(logits, idx);

            // 2. Gate weights: softmax over requantized top-k logits via LUT.
            int32_t qlog[K_];
            int32_t maxq = 0;
            for (std::size_t j = 0; j < K_; ++j)
            {
                qlog[j] = static_cast<int32_t>(router_requantizer.apply(logits[idx[j]]));
                if (j == 0 || qlog[j] > maxq)
                {
                    maxq = qlog[j];
                }
            }
            int64_t expq[K_];
            int64_t sum = 0;
            for (std::size_t j = 0; j < K_; ++j)
            {
                const std::size_t lutIdx = qSoftmaxLUTIndex(qlog[j], maxq);
                expq[j] = static_cast<int64_t>(exp_lut[lutIdx]);
                sum += expq[j];
            }

            int32_t gate[K_];
            const int64_t one = static_cast<int64_t>(1) << kGateShift;
            if (sum <= 0)
            {
                for (std::size_t j = 0; j < K_; ++j)
                {
                    gate[j] = static_cast<int32_t>(one / static_cast<int64_t>(K_));
                }
            }
            else
            {
                const int64_t half = sum >> 1;
                for (std::size_t j = 0; j < K_; ++j)
                {
                    gate[j] = static_cast<int32_t>((expq[j] * one + half) / sum);
                }
            }

            // 3. Run the selected experts and blend over (y - out_zp) in Q15.
            const int32_t out_zp = static_cast<int32_t>(output_zero_point);
            int64_t blend[NumOutputs_];
            for (std::size_t o = 0; o < NumOutputs_; ++o)
            {
                blend[o] = 0;
            }
            OutputType expertOut[NumOutputs_];
            for (std::size_t j = 0; j < K_; ++j)
            {
                experts[idx[j]].forward(input, expertOut);
                for (std::size_t o = 0; o < NumOutputs_; ++o)
                {
                    blend[o] += static_cast<int64_t>(gate[j]) *
                        (static_cast<int64_t>(expertOut[o]) - out_zp);
                }
            }

            const int32_t lo = static_cast<int32_t>(output_qmin);
            const int32_t hi = static_cast<int32_t>(output_qmax);
            const int64_t gate_half = static_cast<int64_t>(1) << (kGateShift - 1);
            for (std::size_t o = 0; o < NumOutputs_; ++o)
            {
                int32_t v = out_zp + static_cast<int32_t>(
                    (blend[o] + gate_half) >> kGateShift);
                if (v < lo) v = lo;
                if (v > hi) v = hi;
                output[o] = static_cast<OutputType>(v);
            }

            return idx[0];
        }

        static void selectTopK(AccumulatorType const* const values, std::size_t* indices)
        {
            bool used[NumExperts_];
            for (std::size_t e = 0; e < NumExperts_; ++e)
            {
                used[e] = false;
            }
            for (std::size_t j = 0; j < K_; ++j)
            {
                std::size_t best = NumExperts_;
                for (std::size_t e = 0; e < NumExperts_; ++e)
                {
                    if (used[e])
                    {
                        continue;
                    }
                    if (best == NumExperts_ || values[e] > values[best])
                    {
                        best = e;
                    }
                }
                indices[j] = best;
                used[best] = true;
            }
        }

        static_assert(NumExperts_ > 0, "Number of experts must be > 0.");
        static_assert(K_ > 0, "K must be > 0.");
        static_assert(K_ <= NumExperts_, "K must be <= NumExperts.");
        static_assert(NumInputs_ > 0, "Number of inputs must be > 0.");
        static_assert(NumOutputs_ > 0, "Number of outputs must be > 0.");
    };

} // namespace tinymind
