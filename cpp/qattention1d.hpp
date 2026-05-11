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

#include <cstddef>
#include <cstdint>

/*
 * Quantized linear self-attention (ReLU kernel feature map).
 *
 * Integer counterpart of cpp/selfattention1d.hpp. Replaces softmax with a
 * ReLU feature map so the score normalization disappears and the layer
 * complexity falls from O(N^2 * D) to O(N * D * P + N * P^2). Math:
 *
 *   Q' = qrelu(X * W_q + b_q)        (S x P, in q_scale)
 *   K' = qrelu(X * W_k + b_k)        (S x P, in k_scale)
 *   V  =       X * W_v + b_v         (S x P, in v_scale)
 *   KV = K'^T * V                    (P x P, in kv_scale)
 *   Y  = Q' * KV                     (S x P, in output_scale)
 *
 * Each MAC is rescaled to the next stage's grid by an ordinary Requantizer
 * (qaffine.hpp); the ReLU on Q'/K' is folded into the projection
 * requantizer's qmin == zero_point, matching the existing fused-clamp
 * pattern in qactivations.hpp.
 *
 * Weights / biases / requantizer constants are caller-owned; scratch
 * buffers for Q', K', V', KV are also caller-owned so a transformer
 * encoder block can amortize them across heads and layers.
 *
 * Pure integer at runtime; freestanding-safe.
 */

namespace tinymind {

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
    struct QAttention1D
    {
        typedef InputStorage_   InputType;
        typedef WeightStorage_  WeightType;
        typedef AccumStorage_   AccumType;
        typedef ProjStorage_    ProjType;
        typedef IntermStorage_  IntermType;
        typedef OutputStorage_  OutputType;

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
        // row-major (input feature outer, projection feature inner).
        const WeightType* weights;
        // Concatenated [b_q | b_k | b_v]; each is [ProjectionDim] int32 in
        // the corresponding projection's scale. May be nullptr to indicate
        // bias-free projections (all three).
        const AccumType*  biases;

        InputType  input_zero_point;
        ProjType   q_zero_point;
        ProjType   k_zero_point;
        ProjType   v_zero_point;
        IntermType kv_zero_point;

        // X @ W_q  -> Q' grid.  qmin = q_zero_point folds the ReLU in.
        Requantizer<AccumType, ProjType>   q_requantizer;
        // X @ W_k  -> K' grid.  qmin = k_zero_point folds the ReLU in.
        Requantizer<AccumType, ProjType>   k_requantizer;
        // X @ W_v  -> V  grid.  No activation; standard qmin/qmax.
        Requantizer<AccumType, ProjType>   v_requantizer;
        // K'^T @ V -> KV grid.
        Requantizer<AccumType, IntermType> kv_requantizer;
        // Q' @ KV  -> output grid.
        Requantizer<AccumType, OutputType> output_requantizer;

        void forward(const InputType* input,
                     ProjType*        q_scratch,
                     ProjType*        k_scratch,
                     ProjType*        v_scratch,
                     IntermType*      kv_scratch,
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

            // Steps 1-3: per-row projections Q, K, V. ReLU on Q/K is folded
            // into the requantizer qmin >= projection zero_point at calibration.
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

            // Step 4: KV = K'^T * V. Inner loop sums over the sequence axis.
            const int32_t k_zp = static_cast<int32_t>(k_zero_point);
            const int32_t v_zp = static_cast<int32_t>(v_zero_point);
            for (std::size_t i = 0; i < ProjectionDim_; ++i)
            {
                for (std::size_t j = 0; j < ProjectionDim_; ++j)
                {
                    AccumType acc = 0;
                    for (std::size_t t = 0; t < SequenceLength_; ++t)
                    {
                        const AccumType k =
                            static_cast<AccumType>(k_scratch[t * ProjectionDim_ + i]) - k_zp;
                        const AccumType v =
                            static_cast<AccumType>(v_scratch[t * ProjectionDim_ + j]) - v_zp;
                        acc += k * v;
                    }
                    kv_scratch[i * ProjectionDim_ + j] =
                        kv_requantizer.apply(acc);
                }
            }

            // Step 5: Y = Q' * KV.
            const int32_t q_zp  = static_cast<int32_t>(q_zero_point);
            const int32_t kv_zp = static_cast<int32_t>(kv_zero_point);
            for (std::size_t t = 0; t < SequenceLength_; ++t)
            {
                const ProjType* q_row = q_scratch + t * ProjectionDim_;
                OutputType*     o_row = output + t * ProjectionDim_;
                for (std::size_t j = 0; j < ProjectionDim_; ++j)
                {
                    AccumType acc = 0;
                    for (std::size_t i = 0; i < ProjectionDim_; ++i)
                    {
                        const AccumType q  =
                            static_cast<AccumType>(q_row[i]) - q_zp;
                        const AccumType kv =
                            static_cast<AccumType>(kv_scratch[i * ProjectionDim_ + j]) - kv_zp;
                        acc += q * kv;
                    }
                    o_row[j] = output_requantizer.apply(acc);
                }
            }
        }

        static_assert(SequenceLength_ > 0, "Sequence length must be > 0.");
        static_assert(EmbeddingDim_   > 0, "Embedding dimension must be > 0.");
        static_assert(ProjectionDim_  > 0, "Projection dimension must be > 0.");
    };

} // namespace tinymind
