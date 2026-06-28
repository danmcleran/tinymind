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
 * Key/Value cache state for autoregressive (decoder) attention.
 *
 * Two cache shapes, one per attention flavor, both freestanding POD with
 * inline storage so a decode loop on an MCU never touches the heap:
 *
 *   QLinearKVState   -- linear (ReLU-kernel) causal attention. The entire
 *                       attended history collapses into a fixed P x P running
 *                       sum of K'^T V (int32 accumulator). Memory is O(P^2)
 *                       and CONSTANT in the sequence length: step t folds the
 *                       new token into the accumulator and the past is gone.
 *                       This is the property that makes linear attention the
 *                       natural decoder primitive for tiny targets.
 *
 *   QSoftmaxKVCache  -- standard (softmax) causal attention. Softmax cannot
 *                       collapse the history, so the per-position K and V
 *                       projections are retained. Memory is O(MaxSeq * P) and
 *                       GROWS one row per emitted token until MaxSeq is hit.
 *
 * Both are also reused by the cross-attention layers (qcrossattention.hpp):
 * there the cache is prefilled once from the encoder memory and then read on
 * every decoder query, rather than appended to step by step.
 *
 * The cache object is caller-owned and explicitly reset() before a new
 * sequence; it carries no scales or requantizers (those live on the attention
 * layer). Pure integer at runtime; freestanding-safe.
 */

namespace tinymind
{

    /**
     * Running K'^T V accumulator for linear causal attention.
     *
     * kv[i * ProjectionDim + j] is the int32 sum over all attended positions
     * s of (K'[s][i] - k_zp) * (V[s][j] - v_zp). A decode step adds the new
     * token's outer product; the output read requantizes this accumulator to
     * the intermediate grid before the Q' * KV product. reset() zeroes the
     * accumulator to start a fresh sequence.
     */
    template<typename AccumStorage_, std::size_t ProjectionDim_>
    struct QLinearKVState
    {
        typedef AccumStorage_ AccumType;

        static constexpr std::size_t ProjectionDim = ProjectionDim_;
        static constexpr std::size_t Size          = ProjectionDim_ * ProjectionDim_;

        AccumType kv[Size];

        void reset()
        {
            for (std::size_t i = 0; i < Size; ++i)
            {
                kv[i] = static_cast<AccumType>(0);
            }
        }

        static_assert(ProjectionDim_ > 0, "Projection dimension must be > 0.");
    };

    /**
     * Per-position K / V cache for softmax causal (and cross) attention.
     *
     * k[s * ProjectionDim + p] / v[s * ProjectionDim + p] hold the int8 (or
     * other ProjStorage) projected key/value for cached position s. length is
     * the number of valid positions; a decode step appends one row and bumps
     * length. The attention read iterates s = 0 .. length - 1, which is the
     * causal mask (positions beyond length simply do not exist yet).
     *
     * append() is a no-op once the cache is full, so a runaway decode loop
     * saturates at MaxSeq rather than overrunning the buffer; callers that
     * need to know should compare length against MaxSeq before stepping.
     */
    template<typename ProjStorage_, std::size_t MaxSeq_, std::size_t ProjectionDim_>
    struct QSoftmaxKVCache
    {
        typedef ProjStorage_ ProjType;

        static constexpr std::size_t MaxSeq        = MaxSeq_;
        static constexpr std::size_t ProjectionDim = ProjectionDim_;
        static constexpr std::size_t Size          = MaxSeq_ * ProjectionDim_;

        ProjType    k[Size];
        ProjType    v[Size];
        std::size_t length;

        void reset()
        {
            length = 0;
        }

        bool full() const
        {
            return length >= MaxSeq_;
        }

        /**
         * Append one projected key/value row. Returns the slot index written,
         * or MaxSeq (an invalid index) if the cache was already full.
         */
        std::size_t append(const ProjType* k_row, const ProjType* v_row)
        {
            if (length >= MaxSeq_)
            {
                return MaxSeq_;
            }
            const std::size_t slot = length;
            for (std::size_t p = 0; p < ProjectionDim_; ++p)
            {
                k[slot * ProjectionDim_ + p] = k_row[p];
                v[slot * ProjectionDim_ + p] = v_row[p];
            }
            ++length;
            return slot;
        }

        static_assert(MaxSeq_        > 0, "Max sequence length must be > 0.");
        static_assert(ProjectionDim_ > 0, "Projection dimension must be > 0.");
    };

} // namespace tinymind
