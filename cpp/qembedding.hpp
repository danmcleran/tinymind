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
 * Quantized token embedding lookup (gather).
 *
 * The first layer of a transformer: maps a sequence of integer token ids
 * into a sequence of int8 embedding vectors by gathering rows from a
 * caller-owned weight table. This is the int8 counterpart of PyTorch's
 * nn.Embedding / a tf.gather over a [VocabSize, EmbeddingDim] matrix.
 *
 * Type contract:
 *
 *   WeightStorage = int8_t   embedding table, asymmetric-affine quantized
 *                            (table_zero_point baked into the optional
 *                            requantizer; symmetric tables use zp = 0)
 *   OutputStorage = int8_t   activation grid handed to the next layer
 *
 * Layout:
 *
 *   table   row-major [VocabSize * EmbeddingDim]
 *   tokens  length seq_len, each in [0, VocabSize)
 *   output  row-major [seq_len * EmbeddingDim]
 *
 * Two operating modes:
 *
 *   requantizer == nullptr : pure gather/copy. Use when the embedding table
 *                            already lives on the OUTPUT activation grid
 *                            (the common case -- the embedding defines the
 *                            grid the rest of the network calibrates to).
 *
 *   requantizer != nullptr : each gathered code is rescaled from the table
 *                            grid onto the output grid. Use when the table
 *                            and the downstream activations were calibrated
 *                            with different (scale, zero_point).
 *
 * Token ids are NOT range-checked at runtime (freestanding hot path). The
 * caller guarantees 0 <= tokens[t] < VocabSize.
 *
 * Pure integer at runtime; no float, no <cmath>, no stdlib. Safe to compile
 * in the freestanding (FLOAT=0, STD=0) configuration.
 */

namespace tinymind {

    template<
        typename WeightStorage_,
        typename OutputStorage_,
        std::size_t VocabSize_,
        std::size_t EmbeddingDim_>
    struct QEmbedding
    {
        typedef WeightStorage_ WeightType;
        typedef OutputStorage_ OutputType;

        static constexpr std::size_t VocabSize = VocabSize_;
        static constexpr std::size_t EmbeddingDim = EmbeddingDim_;

        const WeightType* table;
        const Requantizer<int32_t, OutputType>* requantizer;

        void forward(const int32_t* tokens, std::size_t seq_len,
                     OutputType* output) const
        {
            for (std::size_t t = 0; t < seq_len; ++t)
            {
                const std::size_t id = static_cast<std::size_t>(tokens[t]);
                const WeightType* row = table + id * EmbeddingDim_;
                OutputType* out_row = output + t * EmbeddingDim_;

                if (requantizer != nullptr)
                {
                    for (std::size_t e = 0; e < EmbeddingDim_; ++e)
                    {
                        out_row[e] = requantizer->apply(
                            static_cast<int32_t>(row[e]));
                    }
                }
                else
                {
                    for (std::size_t e = 0; e < EmbeddingDim_; ++e)
                    {
                        out_row[e] = static_cast<OutputType>(row[e]);
                    }
                }
            }
        }
    };

} // namespace tinymind
