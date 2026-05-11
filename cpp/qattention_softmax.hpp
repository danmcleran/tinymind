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

#include <cstddef>
#include <cstdint>

/*
 * Quantized standard (softmax) self-attention.
 *
 * Computes:
 *
 *   Q = X * W_q + b_q   (S x P, in q_scale)
 *   K = X * W_k + b_k   (S x P, in k_scale)
 *   V = X * W_v + b_v   (S x P, in v_scale)
 *   S_ij = (Q[i] . K[j]) / sqrt(P)        (S x S, in score_scale)
 *   A    = softmax(S, axis=-1)            (S x S, scale 1/256, zp = -128)
 *   Y    = A * V                          (S x P, in output_scale)
 *
 * The 1/sqrt(P) scaling factor is folded into score_requantizer so the
 * inference path never sees a square root or a float.
 *
 * Softmax follows the TFLite int8 convention: per-row max subtract, exp
 * LUT lookup, normalize to 1/256 scale at zero_point -128 (the full int8
 * range maps to probability [0, 1]). The exp LUT is built host-side by
 * buildQSoftmaxExpLUT in qcalibration.hpp; the layer holds the pointer.
 *
 * Heavier than the ReLU-kernel QAttention1D because of the S x S score
 * tensor and the LUT pass, so it lives in its own header. Weights /
 * biases / requantizers / scratch buffers are all caller-owned, matching
 * the rest of the q*.hpp family. Pure integer at runtime;
 * freestanding-safe.
 */

namespace tinymind {

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
        std::size_t ProjectionDim_>
    struct QAttentionSoftmax1D
    {
        typedef InputStorage_   InputType;
        typedef WeightStorage_  WeightType;
        typedef AccumStorage_   AccumType;
        typedef ProjStorage_    ProjType;
        typedef ScoreStorage_   ScoreType;
        typedef AttnStorage_    AttnType;
        typedef OutputStorage_  OutputType;

        static constexpr std::size_t SequenceLength       = SequenceLength_;
        static constexpr std::size_t EmbeddingDim         = EmbeddingDim_;
        static constexpr std::size_t ProjectionDim        = ProjectionDim_;
        static constexpr std::size_t InputSize            = SequenceLength_ * EmbeddingDim_;
        static constexpr std::size_t OutputSize           = SequenceLength_ * ProjectionDim_;
        static constexpr std::size_t WeightsPerProjection = EmbeddingDim_ * ProjectionDim_;
        static constexpr std::size_t TotalWeights         = 3 * WeightsPerProjection;
        static constexpr std::size_t TotalBiases          = 3 * ProjectionDim_;

        static constexpr std::size_t QScratchSize     = SequenceLength_ * ProjectionDim_;
        static constexpr std::size_t KScratchSize     = SequenceLength_ * ProjectionDim_;
        static constexpr std::size_t VScratchSize     = SequenceLength_ * ProjectionDim_;
        static constexpr std::size_t ScoreScratchSize = SequenceLength_ * SequenceLength_;
        static constexpr std::size_t AttnScratchSize  = SequenceLength_ * SequenceLength_;

        // Concatenated [W_q | W_k | W_v]; same layout as QAttention1D.
        const WeightType* weights;
        // Concatenated [b_q | b_k | b_v]; may be nullptr.
        const AccumType*  biases;

        InputType  input_zero_point;
        ProjType   q_zero_point;
        ProjType   k_zero_point;
        ProjType   v_zero_point;
        AttnType   attn_zero_point;  // matches softmax output (typically -128)

        // X @ W_x -> projection grids.
        Requantizer<AccumType, ProjType>   q_requantizer;
        Requantizer<AccumType, ProjType>   k_requantizer;
        Requantizer<AccumType, ProjType>   v_requantizer;

        // Q @ K^T -> score grid. The 1/sqrt(ProjectionDim) factor is folded
        // into this requantizer's (multiplier, shift) at calibration time.
        Requantizer<AccumType, ScoreType>  score_requantizer;

        // Softmax LUT (256-entry int32 exp table) and saturation bounds. The
        // attention output convention is TFLite-fixed (scale 1/256, zero
        // point -128); zp is captured by attn_zero_point above.
        const int32_t* softmax_exp_lut;
        AttnType       attn_qmin;
        AttnType       attn_qmax;

        // A @ V -> output grid.
        Requantizer<AccumType, OutputType> output_requantizer;

        void forward(const InputType* input,
                     ProjType*        q_scratch,
                     ProjType*        k_scratch,
                     ProjType*        v_scratch,
                     ScoreType*       score_scratch,
                     AttnType*        attn_scratch,
                     OutputType*      output) const
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

            // Steps 1-3: Q, K, V projections.
            for (std::size_t t = 0; t < SequenceLength_; ++t)
            {
                const InputType* x_row = input + t * EmbeddingDim_;
                ProjType* q_row = q_scratch + t * ProjectionDim_;
                ProjType* k_row = k_scratch + t * ProjectionDim_;
                ProjType* v_row = v_scratch + t * ProjectionDim_;

                for (std::size_t p = 0; p < ProjectionDim_; ++p)
                {
                    AccumType acc_q = (b_q != nullptr)
                        ? b_q[p] : static_cast<AccumType>(0);
                    AccumType acc_k = (b_k != nullptr)
                        ? b_k[p] : static_cast<AccumType>(0);
                    AccumType acc_v = (b_v != nullptr)
                        ? b_v[p] : static_cast<AccumType>(0);

                    for (std::size_t e = 0; e < EmbeddingDim_; ++e)
                    {
                        const AccumType x =
                            static_cast<AccumType>(x_row[e]) - in_zp;
                        acc_q += static_cast<AccumType>(
                            w_q[e * ProjectionDim_ + p]) * x;
                        acc_k += static_cast<AccumType>(
                            w_k[e * ProjectionDim_ + p]) * x;
                        acc_v += static_cast<AccumType>(
                            w_v[e * ProjectionDim_ + p]) * x;
                    }

                    q_row[p] = q_requantizer.apply(acc_q);
                    k_row[p] = k_requantizer.apply(acc_k);
                    v_row[p] = v_requantizer.apply(acc_v);
                }
            }

            // Step 4: scores S_ij = (Q[i] . K[j]) / sqrt(P), requantized to
            // the score grid. 1/sqrt(P) factor lives in score_requantizer.
            const int32_t q_zp = static_cast<int32_t>(q_zero_point);
            const int32_t k_zp = static_cast<int32_t>(k_zero_point);
            for (std::size_t i = 0; i < SequenceLength_; ++i)
            {
                const ProjType* q_row = q_scratch + i * ProjectionDim_;
                ScoreType*      s_row = score_scratch + i * SequenceLength_;
                for (std::size_t j = 0; j < SequenceLength_; ++j)
                {
                    const ProjType* k_row = k_scratch + j * ProjectionDim_;
                    AccumType acc = 0;
                    for (std::size_t p = 0; p < ProjectionDim_; ++p)
                    {
                        const AccumType q =
                            static_cast<AccumType>(q_row[p]) - q_zp;
                        const AccumType k =
                            static_cast<AccumType>(k_row[p]) - k_zp;
                        acc += q * k;
                    }
                    s_row[j] = score_requantizer.apply(acc);
                }
            }

            // Step 5: softmax across the score row -> attention weights.
            // Inlined per-row TFLite-style softmax (mirrors QSoftmax1D). We
            // do not delegate to QSoftmax1D::forward because ScoreType /
            // AttnType may differ from each other and from the QSoftmax1D
            // single-storage template signature.
            const int32_t attn_zp = static_cast<int32_t>(attn_zero_point);
            const int32_t attn_lo = static_cast<int32_t>(attn_qmin);
            const int32_t attn_hi = static_cast<int32_t>(attn_qmax);
            for (std::size_t r = 0; r < SequenceLength_; ++r)
            {
                const ScoreType* s_row = score_scratch + r * SequenceLength_;
                AttnType*        a_row = attn_scratch  + r * SequenceLength_;

                int32_t max_q = static_cast<int32_t>(s_row[0]);
                for (std::size_t j = 1; j < SequenceLength_; ++j)
                {
                    const int32_t v = static_cast<int32_t>(s_row[j]);
                    if (v > max_q) max_q = v;
                }

                int64_t sum = 0;
                for (std::size_t j = 0; j < SequenceLength_; ++j)
                {
                    const std::size_t idx = qSoftmaxLUTIndex(
                        static_cast<int32_t>(s_row[j]), max_q);
                    sum += static_cast<int64_t>(softmax_exp_lut[idx]);
                }

                if (sum <= 0)
                {
                    for (std::size_t j = 0; j < SequenceLength_; ++j)
                    {
                        int32_t q = attn_zp;
                        if (q < attn_lo) q = attn_lo;
                        if (q > attn_hi) q = attn_hi;
                        a_row[j] = static_cast<AttnType>(q);
                    }
                    continue;
                }

                const int64_t half_sum = sum >> 1;
                for (std::size_t j = 0; j < SequenceLength_; ++j)
                {
                    const std::size_t idx = qSoftmaxLUTIndex(
                        static_cast<int32_t>(s_row[j]), max_q);
                    const int64_t numerator =
                        (static_cast<int64_t>(softmax_exp_lut[idx]) << 8) + half_sum;
                    const int32_t scaled =
                        static_cast<int32_t>(numerator / sum);
                    int32_t q = scaled + attn_zp;
                    if (q < attn_lo) q = attn_lo;
                    if (q > attn_hi) q = attn_hi;
                    a_row[j] = static_cast<AttnType>(q);
                }
            }

            // Step 6: Y = A * V.
            const int32_t a_zp = attn_zp;
            const int32_t v_zp = static_cast<int32_t>(v_zero_point);
            for (std::size_t t = 0; t < SequenceLength_; ++t)
            {
                const AttnType* a_row = attn_scratch + t * SequenceLength_;
                OutputType*     o_row = output + t * ProjectionDim_;
                for (std::size_t p = 0; p < ProjectionDim_; ++p)
                {
                    AccumType acc = 0;
                    for (std::size_t s = 0; s < SequenceLength_; ++s)
                    {
                        const AccumType a =
                            static_cast<AccumType>(a_row[s]) - a_zp;
                        const AccumType v =
                            static_cast<AccumType>(v_scratch[s * ProjectionDim_ + p]) - v_zp;
                        acc += a * v;
                    }
                    o_row[p] = output_requantizer.apply(acc);
                }
            }
        }

        static_assert(SequenceLength_ > 0, "Sequence length must be > 0.");
        static_assert(EmbeddingDim_   > 0, "Embedding dimension must be > 0.");
        static_assert(ProjectionDim_  > 0, "Projection dimension must be > 0.");
    };

} // namespace tinymind
