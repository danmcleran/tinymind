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
#include "qaffine.hpp"
#include "qsoftmax.hpp"
#include "qkvcache.hpp"

#include <cstddef>
#include <cstdint>

/*
 * Quantized standard (softmax) CAUSAL self-attention with a growing KV cache.
 *
 * Decoder counterpart of cpp/qattention_softmax.hpp. Computes, for query
 * position t attending only to keys/values at positions s <= t:
 *
 *   Q[t]   = X[t] * W_q + b_q                       (P, in q_scale)
 *   K[t]   = X[t] * W_k + b_k                       (P, in k_scale)  -> cache
 *   V[t]   = X[t] * W_v + b_v                       (P, in v_scale)  -> cache
 *   s_tj   = (Q[t] . K[j]) / sqrt(P), j = 0..t      (in score_scale)
 *   a_t    = softmax(s_t, axis=-1)                  (scale 1/256, zp -128)
 *   Y[t]   = sum_{j <= t} a_tj * V[j]               (P, in output_scale)
 *
 * Unlike linear attention, softmax cannot collapse the attended history, so
 * the per-position K and V projections are retained in a QSoftmaxKVCache.
 * Memory grows one (K, V) row per token until MaxSeq. The causal mask is the
 * cache length itself: the score loop runs j = 0 .. length - 1, so a position
 * physically cannot attend to a token that has not been appended yet.
 *
 * The 1/sqrt(P) factor folds into score_requantizer; softmax follows the
 * TFLite int8 convention via the host-built exp LUT (buildQSoftmaxExpLUT in
 * qcalibration.hpp), exactly as qattention_softmax.hpp does.
 *
 * Two entry points sharing one token kernel:
 *   step()    -- append one token to the cache and emit its output row.
 *   forward() -- reset the cache and run step() across a whole [S x E] block,
 *                byte-identical to S successive step() calls. The full-sequence
 *                pass is the lower-triangular masked attention.
 *
 * Weights / biases / requantizers / scratch are caller-owned. Pure integer at
 * runtime; freestanding-safe (the exp LUT is a plain const int32 table).
 */

namespace tinymind
{

    template<
        typename InputStorage_,
        typename WeightStorage_,
        typename AccumStorage_,
        typename ProjStorage_,
        typename ScoreStorage_,
        typename AttnStorage_,
        typename OutputStorage_,
        std::size_t SequenceLength_,
        std::size_t EmbeddingDim_,
        std::size_t ProjectionDim_,
        std::size_t MaxSeq_ = SequenceLength_>
    struct QCausalAttentionSoftmax1D
    {
        typedef InputStorage_   InputType;
        typedef WeightStorage_  WeightType;
        typedef AccumStorage_   AccumType;
        typedef ProjStorage_    ProjType;
        typedef ScoreStorage_   ScoreType;
        typedef AttnStorage_    AttnType;
        typedef OutputStorage_  OutputType;

        typedef QSoftmaxKVCache<ProjStorage_, MaxSeq_, ProjectionDim_> KVCache;

        static constexpr std::size_t SequenceLength       = SequenceLength_;
        static constexpr std::size_t EmbeddingDim         = EmbeddingDim_;
        static constexpr std::size_t ProjectionDim        = ProjectionDim_;
        static constexpr std::size_t MaxSeq               = MaxSeq_;
        static constexpr std::size_t InputSize            = SequenceLength_ * EmbeddingDim_;
        static constexpr std::size_t OutputSize           = SequenceLength_ * ProjectionDim_;
        static constexpr std::size_t WeightsPerProjection = EmbeddingDim_ * ProjectionDim_;
        static constexpr std::size_t TotalWeights         = 3 * WeightsPerProjection;
        static constexpr std::size_t TotalBiases          = 3 * ProjectionDim_;

        static constexpr std::size_t QScratchSize     = SequenceLength_ * ProjectionDim_;
        static constexpr std::size_t ScoreScratchSize = MaxSeq_;
        static constexpr std::size_t AttnScratchSize  = MaxSeq_;

        // Concatenated [W_q | W_k | W_v]; same layout as QAttentionSoftmax1D.
        const WeightType* weights;
        const AccumType*  biases;

        InputType  input_zero_point;
        ProjType   q_zero_point;
        ProjType   k_zero_point;
        ProjType   v_zero_point;
        AttnType   attn_zero_point;  // softmax output zp (typically -128)

        Requantizer<AccumType, ProjType>   q_requantizer;
        Requantizer<AccumType, ProjType>   k_requantizer;
        Requantizer<AccumType, ProjType>   v_requantizer;
        // Q @ K^T -> score grid; 1/sqrt(P) folded into (multiplier, shift).
        Requantizer<AccumType, ScoreType>  score_requantizer;

        const int32_t* softmax_exp_lut;  // [256]
        AttnType       attn_qmin;
        AttnType       attn_qmax;

        Requantizer<AccumType, OutputType> output_requantizer;

        /**
         * Project one input row into Q, K, V (each ProjectionDim wide). No
         * activation fold (softmax attention keeps the raw projections).
         */
        void project(const InputType* x_row,
                     ProjType*        q_row,
                     ProjType*        k_row,
                     ProjType*        v_row) const
        {
            const int32_t in_zp = static_cast<int32_t>(input_zero_point);

            const WeightType* w_q = weights;
            const WeightType* w_k = weights + WeightsPerProjection;
            const WeightType* w_v = weights + 2 * WeightsPerProjection;
            const AccumType*  b_q = (biases != nullptr)
                ? biases : static_cast<const AccumType*>(nullptr);
            const AccumType*  b_k = (biases != nullptr)
                ? biases + ProjectionDim_ : static_cast<const AccumType*>(nullptr);
            const AccumType*  b_v = (biases != nullptr)
                ? biases + 2 * ProjectionDim_ : static_cast<const AccumType*>(nullptr);

            for (std::size_t p = 0; p < ProjectionDim_; ++p)
            {
                AccumType acc_q = (b_q != nullptr) ? b_q[p] : static_cast<AccumType>(0);
                AccumType acc_k = (b_k != nullptr) ? b_k[p] : static_cast<AccumType>(0);
                AccumType acc_v = (b_v != nullptr) ? b_v[p] : static_cast<AccumType>(0);

                for (std::size_t e = 0; e < EmbeddingDim_; ++e)
                {
                    const AccumType x = static_cast<AccumType>(x_row[e]) - in_zp;
                    acc_q += static_cast<AccumType>(w_q[e * ProjectionDim_ + p]) * x;
                    acc_k += static_cast<AccumType>(w_k[e * ProjectionDim_ + p]) * x;
                    acc_v += static_cast<AccumType>(w_v[e * ProjectionDim_ + p]) * x;
                }

                q_row[p] = q_requantizer.apply(acc_q);
                k_row[p] = k_requantizer.apply(acc_k);
                v_row[p] = v_requantizer.apply(acc_v);
            }
        }

        /**
         * Append one token to the cache and emit its causal-attention output.
         *
         * q_row / k_row / v_row are ProjectionDim scratch. score_scratch and
         * attn_scratch are each at least (cache.length + 1) wide (size MaxSeq
         * is always sufficient). The new token is appended before scoring, so
         * it attends to itself plus all earlier cached positions.
         */
        void step(const InputType* x_row,
                  KVCache&         cache,
                  ProjType*        q_row,
                  ProjType*        k_row,
                  ProjType*        v_row,
                  ScoreType*       score_scratch,
                  AttnType*        attn_scratch,
                  OutputType*      o_row) const
        {
            project(x_row, q_row, k_row, v_row);
            cache.append(k_row, v_row);
            const std::size_t L = cache.length;

            // Scores against every cached key (the causal prefix). 1/sqrt(P)
            // lives in score_requantizer.
            const int32_t q_zp = static_cast<int32_t>(q_zero_point);
            const int32_t k_zp = static_cast<int32_t>(k_zero_point);
            for (std::size_t j = 0; j < L; ++j)
            {
                const ProjType* k_cached = cache.k + j * ProjectionDim_;
                AccumType acc = 0;
                for (std::size_t p = 0; p < ProjectionDim_; ++p)
                {
                    const AccumType q = static_cast<AccumType>(q_row[p]) - q_zp;
                    const AccumType k = static_cast<AccumType>(k_cached[p]) - k_zp;
                    acc += q * k;
                }
                score_scratch[j] = score_requantizer.apply(acc);
            }

            // Softmax across the causal prefix (TFLite-style, mirrors
            // QSoftmax1D / QAttentionSoftmax1D).
            const int32_t attn_zp = static_cast<int32_t>(attn_zero_point);
            const int32_t attn_lo = static_cast<int32_t>(attn_qmin);
            const int32_t attn_hi = static_cast<int32_t>(attn_qmax);

            int32_t max_q = static_cast<int32_t>(score_scratch[0]);
            for (std::size_t j = 1; j < L; ++j)
            {
                const int32_t v = static_cast<int32_t>(score_scratch[j]);
                if (v > max_q) max_q = v;
            }

            int64_t sum = 0;
            for (std::size_t j = 0; j < L; ++j)
            {
                const std::size_t idx = qSoftmaxLUTIndex(
                    static_cast<int32_t>(score_scratch[j]), max_q);
                sum += static_cast<int64_t>(softmax_exp_lut[idx]);
            }

            if (sum <= 0)
            {
                for (std::size_t j = 0; j < L; ++j)
                {
                    int32_t q = attn_zp;
                    if (q < attn_lo) q = attn_lo;
                    if (q > attn_hi) q = attn_hi;
                    attn_scratch[j] = static_cast<AttnType>(q);
                }
            }
            else
            {
                const int64_t half_sum = sum >> 1;
                for (std::size_t j = 0; j < L; ++j)
                {
                    const std::size_t idx = qSoftmaxLUTIndex(
                        static_cast<int32_t>(score_scratch[j]), max_q);
                    const int64_t numerator =
                        (static_cast<int64_t>(softmax_exp_lut[idx]) << 8) + half_sum;
                    const int32_t scaled = static_cast<int32_t>(numerator / sum);
                    int32_t q = scaled + attn_zp;
                    if (q < attn_lo) q = attn_lo;
                    if (q > attn_hi) q = attn_hi;
                    attn_scratch[j] = static_cast<AttnType>(q);
                }
            }

            // Y[t] = sum_{j < L} a_j * V[j].
            const int32_t v_zp = static_cast<int32_t>(v_zero_point);
            for (std::size_t p = 0; p < ProjectionDim_; ++p)
            {
                AccumType acc = 0;
                for (std::size_t j = 0; j < L; ++j)
                {
                    const AccumType a = static_cast<AccumType>(attn_scratch[j]) - attn_zp;
                    const AccumType v =
                        static_cast<AccumType>(cache.v[j * ProjectionDim_ + p]) - v_zp;
                    acc += a * v;
                }
                o_row[p] = output_requantizer.apply(acc);
            }
        }

        /**
         * Full-sequence causal pass. Resets the cache and steps across the
         * block; byte-identical to S successive step() calls. This is the
         * lower-triangular masked softmax attention.
         *
         * q_scratch is [S x P]; score_scratch / attn_scratch are [MaxSeq] and
         * reused each step.
         */
        void forward(const InputType* input,
                     KVCache&         cache,
                     ProjType*        q_scratch,
                     ProjType*        k_scratch,
                     ProjType*        v_scratch,
                     ScoreType*       score_scratch,
                     AttnType*        attn_scratch,
                     OutputType*      output) const
        {
            cache.reset();
            for (std::size_t t = 0; t < SequenceLength_; ++t)
            {
                step(input + t * EmbeddingDim_,
                     cache,
                     q_scratch + t * ProjectionDim_,
                     k_scratch + t * ProjectionDim_,
                     v_scratch + t * ProjectionDim_,
                     score_scratch,
                     attn_scratch,
                     output + t * ProjectionDim_);
            }
        }

        static_assert(SequenceLength_ > 0, "Sequence length must be > 0.");
        static_assert(EmbeddingDim_   > 0, "Embedding dimension must be > 0.");
        static_assert(ProjectionDim_  > 0, "Projection dimension must be > 0.");
        static_assert(MaxSeq_ >= SequenceLength_,
                      "Cache MaxSeq must cover the full-sequence forward pass.");
    };

} // namespace tinymind
