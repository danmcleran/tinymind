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
#include "qadd.hpp"

#include <cstddef>
#include <cstdint>

/*
 * Quantized positional encoding (fused add).
 *
 * Transformers have no recurrence or convolution, so token order is injected
 * by adding a per-position vector to the embedding sequence:
 *
 *   output[t, e] = input[t, e] + table[t, e]
 *
 * The positional table is a caller-owned constant of shape
 * [SeqLen, EmbeddingDim]. It may be the fixed sinusoidal encoding from
 * "Attention Is All You Need" (generate with sinusoidalPositionalTable below)
 * or a learned table imported from a trained model -- the runtime does not
 * care which.
 *
 * The add itself is the TFLite-semantics affine ADD: the input and the table
 * may carry different (scale, zero_point), and the embedded QAdd reconciles
 * them into the output grid. This struct is a thin convenience wrapper that
 * binds the constant second operand (the table) so the call site passes only
 * the running activation.
 *
 * Type contract:
 *
 *   InStorage    = int8_t   embedding activations
 *   TableStorage = int8_t   positional codes (own affine grid)
 *   OutStorage   = int8_t   activation grid handed to the first encoder block
 *
 * Pure integer at runtime; no float, no <cmath>, no stdlib. Safe to compile
 * in the freestanding (FLOAT=0, STD=0) configuration. The host-only
 * sinusoidalPositionalTable() helper is additionally gated on FLOAT && STD.
 */

namespace tinymind {

    template<
        typename InStorage_,
        typename TableStorage_,
        typename OutStorage_,
        std::size_t SeqLen_,
        std::size_t EmbeddingDim_>
    struct QPositionalEncoding1D
    {
        typedef InStorage_ InType;
        typedef TableStorage_ TableType;
        typedef OutStorage_ OutType;

        static constexpr std::size_t SeqLen = SeqLen_;
        static constexpr std::size_t EmbeddingDim = EmbeddingDim_;
        static constexpr std::size_t Size = SeqLen_ * EmbeddingDim_;

        const TableType* table;
        QAdd<InStorage_, TableStorage_, OutStorage_, SeqLen_ * EmbeddingDim_> adder;

        void forward(const InType* input, OutType* output) const
        {
            adder.forward(input, table, output);
        }
    };

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
    /*
     * Generate the fixed sinusoidal positional table (host-only).
     *
     *   table[pos, 2i]   = sin(pos / 10000^(2i/dim))
     *   table[pos, 2i+1] = cos(pos / 10000^(2i/dim))
     *
     * Writes seq_len * dim floats row-major. Quantize the result onto the
     * TableStorage grid (computeAffineParamsAsymmetric + quantizeBuffer) to
     * feed QPositionalEncoding1D::table.
     */
    inline void sinusoidalPositionalTable(std::size_t seq_len, std::size_t dim,
                                          float* out)
    {
        for (std::size_t pos = 0; pos < seq_len; ++pos)
        {
            for (std::size_t i = 0; i < dim; ++i)
            {
                const std::size_t pair = i / 2;
                const float exponent =
                    static_cast<float>(2 * pair) / static_cast<float>(dim);
                const float denom = std::pow(10000.0f, exponent);
                const float angle = static_cast<float>(pos) / denom;
                out[pos * dim + i] =
                    ((i & 1u) == 0u) ? std::sin(angle) : std::cos(angle);
            }
        }
    }
#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD

} // namespace tinymind
