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
 * Quantized CROSS-attention (encoder-decoder) -- linear and softmax flavors.
 *
 * The decoder query stream attends to a fixed encoder memory. Unlike self-
 * attention, the keys and values come from a different tensor than the
 * queries, and there is no causal mask: every decoder position sees the whole
 * encoder output. The crucial efficiency property is that K and V depend only
 * on the (constant) encoder memory, so they are PREFILLED once and then read on
 * every decoder query -- the per-token decode cost is just the Q projection
 * plus one attention read.
 *
 *   Q'[t] = proj_q(dec[t])              from the decoder hidden state
 *   K, V  = proj_kv(memory)            from the encoder output  (prefilled)
 *   Y[t]  = attend(Q'[t], K, V)
 *
 * Two layers mirroring the self-attention pair:
 *   QCrossAttention1D         -- linear (ReLU-kernel) cross-attention. The
 *                                encoder K/V collapse into a single P x P KV
 *                                matrix at prefill; each query is one Q' * KV.
 *   QCrossAttentionSoftmax1D  -- standard softmax cross-attention. Prefill
 *                                fills a QSoftmaxKVCache with the encoder K/V
 *                                rows; each query scores against all of them,
 *                                softmaxes (no mask), and reads V.
 *
 * Weight layout is the familiar concatenation [W_q | W_k | W_v]; W_q is applied
 * to decoder rows and W_k / W_v to encoder rows, all [EmbeddingDim x
 * ProjectionDim] (encoder and decoder share d_model, as in a standard
 * transformer). Weights / biases / requantizers / scratch are caller-owned.
 * Pure integer at runtime; freestanding-safe (softmax flavor uses the same
 * host-built exp LUT as the rest of the family).
 */

namespace tinymind
{

    // ----------------------------------------------------------------------
    // Linear (ReLU-kernel) cross-attention.
    // ----------------------------------------------------------------------
    template<
        typename InputStorage_,
        typename WeightStorage_,
        typename AccumStorage_,
        typename ProjStorage_,
        typename IntermStorage_,
        typename OutputStorage_,
        std::size_t DecSeqLength_,
        std::size_t EncSeqLength_,
        std::size_t EmbeddingDim_,
        std::size_t ProjectionDim_>
    struct QCrossAttention1D
    {
        typedef InputStorage_   InputType;
        typedef WeightStorage_  WeightType;
        typedef AccumStorage_   AccumType;
        typedef ProjStorage_    ProjType;
        typedef IntermStorage_  IntermType;
        typedef OutputStorage_  OutputType;

        static constexpr std::size_t DecSeqLength        = DecSeqLength_;
        static constexpr std::size_t EncSeqLength        = EncSeqLength_;
        static constexpr std::size_t EmbeddingDim        = EmbeddingDim_;
        static constexpr std::size_t ProjectionDim       = ProjectionDim_;
        static constexpr std::size_t WeightsPerProjection = EmbeddingDim_ * ProjectionDim_;
        static constexpr std::size_t TotalWeights        = 3 * WeightsPerProjection;
        static constexpr std::size_t TotalBiases         = 3 * ProjectionDim_;

        static constexpr std::size_t KScratchSize  = EncSeqLength_ * ProjectionDim_;
        static constexpr std::size_t VScratchSize  = EncSeqLength_ * ProjectionDim_;
        static constexpr std::size_t KVScratchSize = ProjectionDim_ * ProjectionDim_;
        static constexpr std::size_t QScratchSize  = DecSeqLength_ * ProjectionDim_;

        const WeightType* weights;  // [W_q | W_k | W_v]
        const AccumType*  biases;   // [b_q | b_k | b_v] or nullptr

        // Cross-attention reads two different affine grids: Q from the decoder
        // hidden state, K/V from the encoder memory. Each carries its own input
        // zero point (the matching input scales live in the requantizers).
        InputType  q_input_zero_point;
        InputType  kv_input_zero_point;
        ProjType   q_zero_point;
        ProjType   k_zero_point;
        ProjType   v_zero_point;
        IntermType kv_zero_point;

        Requantizer<AccumType, ProjType>   q_requantizer;  // qmin = q_zp folds ReLU
        Requantizer<AccumType, ProjType>   k_requantizer;  // qmin = k_zp folds ReLU
        Requantizer<AccumType, ProjType>   v_requantizer;
        Requantizer<AccumType, IntermType> kv_requantizer;
        Requantizer<AccumType, OutputType> output_requantizer;

        // Project one encoder row into K', V'. ReLU on K' folded into qmin.
        void projectKV(const InputType* m_row, ProjType* k_row, ProjType* v_row) const
        {
            const int32_t in_zp = static_cast<int32_t>(kv_input_zero_point);
            const WeightType* w_k = weights + WeightsPerProjection;
            const WeightType* w_v = weights + 2 * WeightsPerProjection;
            const AccumType*  b_k = (biases != nullptr)
                ? biases + ProjectionDim_ : static_cast<const AccumType*>(nullptr);
            const AccumType*  b_v = (biases != nullptr)
                ? biases + 2 * ProjectionDim_ : static_cast<const AccumType*>(nullptr);

            for (std::size_t p = 0; p < ProjectionDim_; ++p)
            {
                AccumType acc_k = (b_k != nullptr) ? b_k[p] : static_cast<AccumType>(0);
                AccumType acc_v = (b_v != nullptr) ? b_v[p] : static_cast<AccumType>(0);
                for (std::size_t e = 0; e < EmbeddingDim_; ++e)
                {
                    const AccumType m = static_cast<AccumType>(m_row[e]) - in_zp;
                    acc_k += static_cast<AccumType>(w_k[e * ProjectionDim_ + p]) * m;
                    acc_v += static_cast<AccumType>(w_v[e * ProjectionDim_ + p]) * m;
                }
                k_row[p] = k_requantizer.apply(acc_k);
                v_row[p] = v_requantizer.apply(acc_v);
            }
        }

        // Project one decoder row into Q'. ReLU folded into qmin.
        void projectQ(const InputType* d_row, ProjType* q_row) const
        {
            const int32_t in_zp = static_cast<int32_t>(q_input_zero_point);
            const WeightType* w_q = weights;
            const AccumType*  b_q = (biases != nullptr)
                ? biases : static_cast<const AccumType*>(nullptr);

            for (std::size_t p = 0; p < ProjectionDim_; ++p)
            {
                AccumType acc_q = (b_q != nullptr) ? b_q[p] : static_cast<AccumType>(0);
                for (std::size_t e = 0; e < EmbeddingDim_; ++e)
                {
                    const AccumType d = static_cast<AccumType>(d_row[e]) - in_zp;
                    acc_q += static_cast<AccumType>(w_q[e * ProjectionDim_ + p]) * d;
                }
                q_row[p] = q_requantizer.apply(acc_q);
            }
        }

        /**
         * Prefill the P x P KV matrix from the encoder memory [EncSeq x E].
         * k_scratch / v_scratch are [EncSeq x P]; kv is [P x P]. Run once per
         * encoder output, then call query() per decoder token.
         */
        void prefill(const InputType* memory,
                     ProjType*        k_scratch,
                     ProjType*        v_scratch,
                     IntermType*      kv) const
        {
            for (std::size_t s = 0; s < EncSeqLength_; ++s)
            {
                projectKV(memory + s * EmbeddingDim_,
                          k_scratch + s * ProjectionDim_,
                          v_scratch + s * ProjectionDim_);
            }

            const int32_t k_zp = static_cast<int32_t>(k_zero_point);
            const int32_t v_zp = static_cast<int32_t>(v_zero_point);
            for (std::size_t i = 0; i < ProjectionDim_; ++i)
            {
                for (std::size_t j = 0; j < ProjectionDim_; ++j)
                {
                    AccumType acc = 0;
                    for (std::size_t s = 0; s < EncSeqLength_; ++s)
                    {
                        const AccumType k =
                            static_cast<AccumType>(k_scratch[s * ProjectionDim_ + i]) - k_zp;
                        const AccumType v =
                            static_cast<AccumType>(v_scratch[s * ProjectionDim_ + j]) - v_zp;
                        acc += k * v;
                    }
                    kv[i * ProjectionDim_ + j] = kv_requantizer.apply(acc);
                }
            }
        }

        /**
         * Cross-attend one decoder row against the prefilled KV matrix.
         * q_row is [P] scratch; o_row is [P].
         */
        void query(const InputType*  d_row,
                   const IntermType* kv,
                   ProjType*         q_row,
                   OutputType*       o_row) const
        {
            projectQ(d_row, q_row);

            const int32_t q_zp  = static_cast<int32_t>(q_zero_point);
            const int32_t kv_zp = static_cast<int32_t>(kv_zero_point);
            for (std::size_t j = 0; j < ProjectionDim_; ++j)
            {
                AccumType acc = 0;
                for (std::size_t i = 0; i < ProjectionDim_; ++i)
                {
                    const AccumType q  = static_cast<AccumType>(q_row[i]) - q_zp;
                    const AccumType kv_ij =
                        static_cast<AccumType>(kv[i * ProjectionDim_ + j]) - kv_zp;
                    acc += q * kv_ij;
                }
                o_row[j] = output_requantizer.apply(acc);
            }
        }

        /**
         * Convenience: prefill from memory then cross-attend every decoder row.
         * dec is [DecSeq x E], output is [DecSeq x P]. q_scratch is [DecSeq x P]
         * (or [P], reused) -- here sized per row for clarity.
         */
        void forward(const InputType* dec,
                     const InputType* memory,
                     ProjType*        k_scratch,
                     ProjType*        v_scratch,
                     IntermType*      kv,
                     ProjType*        q_scratch,
                     OutputType*      output) const
        {
            prefill(memory, k_scratch, v_scratch, kv);
            for (std::size_t t = 0; t < DecSeqLength_; ++t)
            {
                query(dec + t * EmbeddingDim_, kv,
                      q_scratch + t * ProjectionDim_,
                      output + t * ProjectionDim_);
            }
        }

        static_assert(DecSeqLength_  > 0, "Decoder sequence length must be > 0.");
        static_assert(EncSeqLength_  > 0, "Encoder sequence length must be > 0.");
        static_assert(EmbeddingDim_  > 0, "Embedding dimension must be > 0.");
        static_assert(ProjectionDim_ > 0, "Projection dimension must be > 0.");
    };

    // ----------------------------------------------------------------------
    // Softmax cross-attention.
    // ----------------------------------------------------------------------
    template<
        typename InputStorage_,
        typename WeightStorage_,
        typename AccumStorage_,
        typename ProjStorage_,
        typename ScoreStorage_,
        typename AttnStorage_,
        typename OutputStorage_,
        std::size_t DecSeqLength_,
        std::size_t EncSeqLength_,
        std::size_t EmbeddingDim_,
        std::size_t ProjectionDim_>
    struct QCrossAttentionSoftmax1D
    {
        typedef InputStorage_   InputType;
        typedef WeightStorage_  WeightType;
        typedef AccumStorage_   AccumType;
        typedef ProjStorage_    ProjType;
        typedef ScoreStorage_   ScoreType;
        typedef AttnStorage_    AttnType;
        typedef OutputStorage_  OutputType;

        typedef QSoftmaxKVCache<ProjStorage_, EncSeqLength_, ProjectionDim_> KVCache;

        static constexpr std::size_t DecSeqLength        = DecSeqLength_;
        static constexpr std::size_t EncSeqLength        = EncSeqLength_;
        static constexpr std::size_t EmbeddingDim        = EmbeddingDim_;
        static constexpr std::size_t ProjectionDim       = ProjectionDim_;
        static constexpr std::size_t WeightsPerProjection = EmbeddingDim_ * ProjectionDim_;
        static constexpr std::size_t TotalWeights        = 3 * WeightsPerProjection;
        static constexpr std::size_t TotalBiases         = 3 * ProjectionDim_;

        static constexpr std::size_t QScratchSize     = DecSeqLength_ * ProjectionDim_;
        static constexpr std::size_t ScoreScratchSize = EncSeqLength_;
        static constexpr std::size_t AttnScratchSize  = EncSeqLength_;

        const WeightType* weights;  // [W_q | W_k | W_v]
        const AccumType*  biases;

        // Q from the decoder grid, K/V from the encoder-memory grid -- each
        // with its own input zero point (input scales live in the requantizers).
        InputType  q_input_zero_point;
        InputType  kv_input_zero_point;
        ProjType   q_zero_point;
        ProjType   k_zero_point;
        ProjType   v_zero_point;
        AttnType   attn_zero_point;

        Requantizer<AccumType, ProjType>   q_requantizer;
        Requantizer<AccumType, ProjType>   k_requantizer;
        Requantizer<AccumType, ProjType>   v_requantizer;
        Requantizer<AccumType, ScoreType>  score_requantizer;  // 1/sqrt(P) folded

        const int32_t* softmax_exp_lut;
        AttnType       attn_qmin;
        AttnType       attn_qmax;

        Requantizer<AccumType, OutputType> output_requantizer;

        void projectKV(const InputType* m_row, ProjType* k_row, ProjType* v_row) const
        {
            const int32_t in_zp = static_cast<int32_t>(kv_input_zero_point);
            const WeightType* w_k = weights + WeightsPerProjection;
            const WeightType* w_v = weights + 2 * WeightsPerProjection;
            const AccumType*  b_k = (biases != nullptr)
                ? biases + ProjectionDim_ : static_cast<const AccumType*>(nullptr);
            const AccumType*  b_v = (biases != nullptr)
                ? biases + 2 * ProjectionDim_ : static_cast<const AccumType*>(nullptr);

            for (std::size_t p = 0; p < ProjectionDim_; ++p)
            {
                AccumType acc_k = (b_k != nullptr) ? b_k[p] : static_cast<AccumType>(0);
                AccumType acc_v = (b_v != nullptr) ? b_v[p] : static_cast<AccumType>(0);
                for (std::size_t e = 0; e < EmbeddingDim_; ++e)
                {
                    const AccumType m = static_cast<AccumType>(m_row[e]) - in_zp;
                    acc_k += static_cast<AccumType>(w_k[e * ProjectionDim_ + p]) * m;
                    acc_v += static_cast<AccumType>(w_v[e * ProjectionDim_ + p]) * m;
                }
                k_row[p] = k_requantizer.apply(acc_k);
                v_row[p] = v_requantizer.apply(acc_v);
            }
        }

        void projectQ(const InputType* d_row, ProjType* q_row) const
        {
            const int32_t in_zp = static_cast<int32_t>(q_input_zero_point);
            const WeightType* w_q = weights;
            const AccumType*  b_q = (biases != nullptr)
                ? biases : static_cast<const AccumType*>(nullptr);

            for (std::size_t p = 0; p < ProjectionDim_; ++p)
            {
                AccumType acc_q = (b_q != nullptr) ? b_q[p] : static_cast<AccumType>(0);
                for (std::size_t e = 0; e < EmbeddingDim_; ++e)
                {
                    const AccumType d = static_cast<AccumType>(d_row[e]) - in_zp;
                    acc_q += static_cast<AccumType>(w_q[e * ProjectionDim_ + p]) * d;
                }
                q_row[p] = q_requantizer.apply(acc_q);
            }
        }

        /**
         * Prefill the K/V cache from the encoder memory [EncSeq x E]. Fills
         * cache to length EncSeq. Run once per encoder output.
         */
        void prefill(const InputType* memory, KVCache& cache) const
        {
            cache.reset();
            ProjType k_row[ProjectionDim_];
            ProjType v_row[ProjectionDim_];
            for (std::size_t s = 0; s < EncSeqLength_; ++s)
            {
                projectKV(memory + s * EmbeddingDim_, k_row, v_row);
                cache.append(k_row, v_row);
            }
        }

        /**
         * Cross-attend one decoder row against the prefilled encoder K/V.
         * No causal mask: the query sees every encoder position. q_row is [P]
         * scratch; score_scratch / attn_scratch are [EncSeq]; o_row is [P].
         */
        void query(const InputType* d_row,
                   const KVCache&   cache,
                   ProjType*        q_row,
                   ScoreType*       score_scratch,
                   AttnType*        attn_scratch,
                   OutputType*      o_row) const
        {
            projectQ(d_row, q_row);
            const std::size_t L = cache.length;

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

        void forward(const InputType* dec,
                     const InputType* memory,
                     KVCache&         cache,
                     ProjType*        q_scratch,
                     ScoreType*       score_scratch,
                     AttnType*        attn_scratch,
                     OutputType*      output) const
        {
            prefill(memory, cache);
            for (std::size_t t = 0; t < DecSeqLength_; ++t)
            {
                query(dec + t * EmbeddingDim_, cache,
                      q_scratch + t * ProjectionDim_,
                      score_scratch, attn_scratch,
                      output + t * ProjectionDim_);
            }
        }

        static_assert(DecSeqLength_  > 0, "Decoder sequence length must be > 0.");
        static_assert(EncSeqLength_  > 0, "Encoder sequence length must be > 0.");
        static_assert(EmbeddingDim_  > 0, "Embedding dimension must be > 0.");
        static_assert(ProjectionDim_ > 0, "Projection dimension must be > 0.");
    };

} // namespace tinymind
