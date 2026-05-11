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
#include "qattention1d.hpp"

#include <cstddef>
#include <cstdint>

/*
 * Multi-head linear attention wrapper.
 *
 * Holds NumHeads independent QAttention1D heads, each calibrated and
 * weighted separately, and stacks their per-head outputs along the
 * projection axis:
 *
 *   output[t, h * HeadDim + p] = heads[h](input)[t, p]
 *
 * The wrapper does not own any weight or requantizer state; each
 * heads[h] carries its own pointers and (multiplier, shift) constants.
 * Scratch buffers (Q', K', V', KV, head_output) are caller-owned and are
 * reused across heads inside forward() since heads run sequentially.
 *
 * This is the linear-attention variant; softmax MHA can be assembled
 * analogously by templating on QAttentionSoftmax1D, but the per-head
 * scratch shape differs (S x S score buffer) so it lives outside this
 * header to keep the simpler case obvious.
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
        std::size_t HeadDim_,
        std::size_t NumHeads_>
    struct QMultiHeadLinearAttention1D
    {
        typedef QAttention1D<
            InputStorage_, WeightStorage_, AccumStorage_,
            ProjStorage_, IntermStorage_, OutputStorage_,
            SequenceLength_, EmbeddingDim_, HeadDim_> HeadType;

        typedef InputStorage_  InputType;
        typedef OutputStorage_ OutputType;
        typedef ProjStorage_   ProjType;
        typedef IntermStorage_ IntermType;

        static constexpr std::size_t SequenceLength = SequenceLength_;
        static constexpr std::size_t EmbeddingDim   = EmbeddingDim_;
        static constexpr std::size_t HeadDim        = HeadDim_;
        static constexpr std::size_t NumHeads       = NumHeads_;
        static constexpr std::size_t TotalDim       = HeadDim_ * NumHeads_;
        static constexpr std::size_t OutputSize     = SequenceLength_ * TotalDim;

        // Scratch buffer sizes a caller must provide; reused per-head.
        static constexpr std::size_t QScratchSize       = HeadType::QScratchSize;
        static constexpr std::size_t KScratchSize       = HeadType::KScratchSize;
        static constexpr std::size_t VScratchSize       = HeadType::VScratchSize;
        static constexpr std::size_t KVScratchSize      = HeadType::KVScratchSize;
        static constexpr std::size_t HeadOutScratchSize = SequenceLength_ * HeadDim_;

        HeadType heads[NumHeads_];

        void forward(const InputType* input,
                     ProjType*        q_scratch,
                     ProjType*        k_scratch,
                     ProjType*        v_scratch,
                     IntermType*      kv_scratch,
                     OutputType*      head_out_scratch,
                     OutputType*      output) const
        {
            for (std::size_t h = 0; h < NumHeads_; ++h)
            {
                heads[h].forward(input,
                                 q_scratch, k_scratch, v_scratch,
                                 kv_scratch, head_out_scratch);

                for (std::size_t t = 0; t < SequenceLength_; ++t)
                {
                    OutputType*       out_row = output + t * TotalDim;
                    const OutputType* src     = head_out_scratch + t * HeadDim_;
                    OutputType*       dst     = out_row + h * HeadDim_;
                    for (std::size_t p = 0; p < HeadDim_; ++p)
                    {
                        dst[p] = src[p];
                    }
                }
            }
        }

        static_assert(NumHeads_ > 0, "NumHeads must be > 0.");
    };

} // namespace tinymind
