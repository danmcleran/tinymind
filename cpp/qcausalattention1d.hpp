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
#include "qkvcache.hpp"

#include <cstddef>
#include <cstdint>

/*
 * Quantized linear CAUSAL self-attention (ReLU kernel feature map).
 *
 * Decoder counterpart of the bidirectional cpp/qattention1d.hpp. The math is
 * identical except that position t attends only to positions s <= t, which
 * for a linear (kernel) attention is exactly a running prefix sum of the KV
 * outer products:
 *
 *   Q'[t] = qrelu(X[t] * W_q + b_q)              (P, in q_scale)
 *   K'[t] = qrelu(X[t] * W_k + b_k)              (P, in k_scale)
 *   V[t]  =       X[t] * W_v + b_v               (P, in v_scale)
 *   KV[t] = sum_{s <= t} K'[s]^T * V[s]          (P x P, running int32 accum)
 *   Y[t]  = Q'[t] * requant(KV[t])               (P, in output_scale)
 *
 * Because the causal mask collapses into the prefix accumulator, the decode
 * state is a fixed P x P int32 matrix (QLinearKVState) that is CONSTANT in
 * the sequence length -- there is no growing KV cache. This is the structural
 * advantage of linear attention on a memory-bounded target.
 *
 * Two entry points sharing one token kernel:
 *   step()    -- fold one new token into the cache and emit its output row.
 *                The autoregressive decode primitive: O(P^2) work and O(1)
 *                extra memory per token.
 *   forward() -- reset the state and run step() across a whole [S x E] block.
 *                Byte-identical to S successive step() calls, so a training-
 *                time full-sequence pass and an inference-time incremental
 *                decode produce the same int8 stream.
 *
 * The ReLU on Q'/K' is folded into the projection requantizer's
 * qmin == zero_point, matching qattention1d.hpp. Weights / biases /
 * requantizer constants and all scratch are caller-owned. Pure integer at
 * runtime; freestanding-safe (no LUT).
 */

namespace tinymind
{

    template<
        typename InputStorage_,
        typename WeightStorage_,
        typename AccumStorage_,
        typename ProjStorage_,
        typename IntermStorage_,
        typename OutputStorage_,
        std::size_t SequenceLength_,
        std::size_t EmbeddingDim_,
        std::size_t ProjectionDim_>
    struct QCausalAttention1D
    {
        typedef InputStorage_   InputType;
        typedef WeightStorage_  WeightType;
        typedef AccumStorage_   AccumType;
        typedef ProjStorage_    ProjType;
        typedef IntermStorage_  IntermType;
        typedef OutputStorage_  OutputType;

        typedef QLinearKVState<AccumStorage_, ProjectionDim_> KVState;

        static constexpr std::size_t SequenceLength       = SequenceLength_;
        static constexpr std::size_t EmbeddingDim         = EmbeddingDim_;
        static constexpr std::size_t ProjectionDim        = ProjectionDim_;
        static constexpr std::size_t InputSize            = SequenceLength_ * EmbeddingDim_;
        static constexpr std::size_t OutputSize           = SequenceLength_ * ProjectionDim_;
        static constexpr std::size_t WeightsPerProjection = EmbeddingDim_ * ProjectionDim_;
        static constexpr std::size_t TotalWeights         = 3 * WeightsPerProjection;
        static constexpr std::size_t TotalBiases          = 3 * ProjectionDim_;

        static constexpr std::size_t QScratchSize  = SequenceLength_ * ProjectionDim_;
        static constexpr std::size_t KScratchSize  = SequenceLength_ * ProjectionDim_;
        static constexpr std::size_t VScratchSize  = SequenceLength_ * ProjectionDim_;
        static constexpr std::size_t KVScratchSize = ProjectionDim_  * ProjectionDim_;

        // Concatenated [W_q | W_k | W_v]; each is [EmbeddingDim x ProjectionDim]
        // row-major. Same layout as QAttention1D.
        const WeightType* weights;
        // Concatenated [b_q | b_k | b_v]; each is [ProjectionDim] int32 in the
        // corresponding projection's scale. May be nullptr (bias-free).
        const AccumType*  biases;

        InputType  input_zero_point;
        ProjType   q_zero_point;
        ProjType   k_zero_point;
        ProjType   v_zero_point;
        IntermType kv_zero_point;

        Requantizer<AccumType, ProjType>   q_requantizer;  // qmin = q_zp folds ReLU
        Requantizer<AccumType, ProjType>   k_requantizer;  // qmin = k_zp folds ReLU
        Requantizer<AccumType, ProjType>   v_requantizer;
        Requantizer<AccumType, IntermType> kv_requantizer;
        Requantizer<AccumType, OutputType> output_requantizer;

        /**
         * Project one input row into Q', K', V (each ProjectionDim wide). The
         * ReLU on Q'/K' is folded into the requantizer qmin at calibration.
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
         * Fold one token into the running KV state and emit its output row.
         *
         * q_row / k_row / v_row are ProjectionDim scratch (the projections for
         * this token). kv_scratch is ProjectionDim^2 scratch holding the
         * requantized current accumulator. state carries the prefix sum across
         * calls and must be reset() before the first token of a sequence.
         */
        void step(const InputType* x_row,
                  KVState&         state,
                  ProjType*        q_row,
                  ProjType*        k_row,
                  ProjType*        v_row,
                  IntermType*      kv_scratch,
                  OutputType*      o_row) const
        {
            project(x_row, q_row, k_row, v_row);

            // Fold the new token's outer product into the prefix accumulator.
            const int32_t k_zp = static_cast<int32_t>(k_zero_point);
            const int32_t v_zp = static_cast<int32_t>(v_zero_point);
            for (std::size_t i = 0; i < ProjectionDim_; ++i)
            {
                const AccumType k = static_cast<AccumType>(k_row[i]) - k_zp;
                for (std::size_t j = 0; j < ProjectionDim_; ++j)
                {
                    const AccumType v = static_cast<AccumType>(v_row[j]) - v_zp;
                    state.kv[i * ProjectionDim_ + j] += k * v;
                }
            }

            // Requantize the current accumulator to the intermediate grid.
            for (std::size_t i = 0; i < ProjectionDim_; ++i)
            {
                for (std::size_t j = 0; j < ProjectionDim_; ++j)
                {
                    kv_scratch[i * ProjectionDim_ + j] =
                        kv_requantizer.apply(state.kv[i * ProjectionDim_ + j]);
                }
            }

            // Y[t] = Q'[t] * KV[t].
            const int32_t q_zp  = static_cast<int32_t>(q_zero_point);
            const int32_t kv_zp = static_cast<int32_t>(kv_zero_point);
            for (std::size_t j = 0; j < ProjectionDim_; ++j)
            {
                AccumType acc = 0;
                for (std::size_t i = 0; i < ProjectionDim_; ++i)
                {
                    const AccumType q  = static_cast<AccumType>(q_row[i]) - q_zp;
                    const AccumType kv =
                        static_cast<AccumType>(kv_scratch[i * ProjectionDim_ + j]) - kv_zp;
                    acc += q * kv;
                }
                o_row[j] = output_requantizer.apply(acc);
            }
        }

        /**
         * Full-sequence causal pass. Resets state and steps across the block;
         * byte-identical to S successive step() calls on the same state.
         *
         * q_scratch / k_scratch / v_scratch are [S x P]; kv_scratch is
         * [P x P] and reused each step.
         */
        void forward(const InputType* input,
                     KVState&         state,
                     ProjType*        q_scratch,
                     ProjType*        k_scratch,
                     ProjType*        v_scratch,
                     IntermType*      kv_scratch,
                     OutputType*      output) const
        {
            state.reset();
            for (std::size_t t = 0; t < SequenceLength_; ++t)
            {
                step(input + t * EmbeddingDim_,
                     state,
                     q_scratch + t * ProjectionDim_,
                     k_scratch + t * ProjectionDim_,
                     v_scratch + t * ProjectionDim_,
                     kv_scratch,
                     output + t * ProjectionDim_);
            }
        }

        static_assert(SequenceLength_ > 0, "Sequence length must be > 0.");
        static_assert(EmbeddingDim_   > 0, "Embedding dimension must be > 0.");
        static_assert(ProjectionDim_  > 0, "Projection dimension must be > 0.");
    };

} // namespace tinymind
