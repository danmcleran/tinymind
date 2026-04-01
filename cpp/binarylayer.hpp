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

#include <cstddef>
#include <cstdint>

#include "nnproperties.hpp"

namespace tinymind {

    namespace detail {
        static const size_t BITS_PER_WORD = 32;

        inline size_t packedWords(const size_t numBits)
        {
            return (numBits + BITS_PER_WORD - 1) / BITS_PER_WORD;
        }

        inline bool getBit(const uint32_t* packed, const size_t index)
        {
            return (packed[index / BITS_PER_WORD] >> (index % BITS_PER_WORD)) & 1u;
        }

        inline void setBit(uint32_t* packed, const size_t index, const bool value)
        {
            const size_t word = index / BITS_PER_WORD;
            const size_t bit = index % BITS_PER_WORD;
            if (value)
            {
                packed[word] |= (1u << bit);
            }
            else
            {
                packed[word] &= ~(1u << bit);
            }
        }

        inline uint32_t popcount32(uint32_t x)
        {
            x = x - ((x >> 1) & 0x55555555u);
            x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
            return (((x + (x >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24;
        }
    } // namespace detail

    /**
     * Binary dense layer: weights and activations constrained to {-1, +1}.
     *
     * Multiplications become XNOR and accumulations become popcount,
     * providing extreme efficiency on systems without an FPU.
     *
     * Weights are stored as packed bits (1 bit per weight) in uint32_t
     * arrays, giving 32x memory reduction compared to full-precision.
     * A real-valued bias per output neuron is maintained for expressiveness.
     *
     * Training uses the straight-through estimator (STE): gradients pass
     * through the sign() binarization as if it were the identity function,
     * with gradients clipped to zero outside [-1, +1].
     *
     * Real-valued "latent" weights are maintained during training. The
     * binary weights used in the forward pass are sign(latent_weight).
     * After training, only the packed binary weights and biases are needed
     * for inference.
     *
     * @tparam ValueType    Numeric type for biases and latent weights (QValue or float/double)
     * @tparam InputSize    Number of input elements
     * @tparam OutputSize   Number of output neurons
     */
    template<
        typename ValueType,
        size_t InputSize,
        size_t OutputSize>
    class BinaryDense
    {
    public:
        static const size_t TotalBinaryWeights = InputSize * OutputSize;
        static const size_t PackedWeightWords = (TotalBinaryWeights + detail::BITS_PER_WORD - 1) / detail::BITS_PER_WORD;
        static const size_t PackedInputWords = (InputSize + detail::BITS_PER_WORD - 1) / detail::BITS_PER_WORD;

        BinaryDense()
        {
            for (size_t i = 0; i < PackedWeightWords; ++i)
            {
                mPackedWeights[i] = 0;
            }
            for (size_t i = 0; i < OutputSize; ++i)
            {
                mBias[i] = zero();
                mBiasGradients[i] = zero();
            }
            for (size_t i = 0; i < TotalBinaryWeights; ++i)
            {
                mLatentWeights[i] = zero();
                mLatentGradients[i] = zero();
            }
        }

        /**
         * Forward pass using XNOR + popcount.
         *
         * Binarizes input via sign(x), then computes the binary dot product
         * with packed weights using XNOR and popcount. The result is scaled
         * and the real-valued bias is added.
         *
         * output[j] = bias[j] + sum_i( sign(input[i]) XNOR weight[j][i] ) mapped to {-1,+1}
         *
         * The popcount gives the number of matching bits. The dot product
         * in {-1,+1} is: dot = 2 * popcount(XNOR(a,b)) - N
         *
         * @param input  Array of InputSize values
         * @param output Array of OutputSize values
         */
        void forward(ValueType const* const input, ValueType* output) const
        {
            // Binarize input: pack sign bits
            uint32_t packedInput[PackedInputWords];
            for (size_t i = 0; i < PackedInputWords; ++i)
            {
                packedInput[i] = 0;
            }
            for (size_t i = 0; i < InputSize; ++i)
            {
                detail::setBit(packedInput, i, !(input[i] < zero()));
            }

            // For each output neuron, compute XNOR-popcount dot product
            for (size_t j = 0; j < OutputSize; ++j)
            {
                const size_t weightBitOffset = j * InputSize;
                const size_t weightWordOffset = weightBitOffset / detail::BITS_PER_WORD;
                const size_t bitShift = weightBitOffset % detail::BITS_PER_WORD;

                int32_t matchCount = 0;

                if (bitShift == 0)
                {
                    // Aligned case: weight row starts at a word boundary
                    const size_t fullWords = InputSize / detail::BITS_PER_WORD;
                    const size_t remainingBits = InputSize % detail::BITS_PER_WORD;

                    for (size_t w = 0; w < fullWords; ++w)
                    {
                        const uint32_t xnorResult = ~(packedInput[w] ^ mPackedWeights[weightWordOffset + w]);
                        matchCount += static_cast<int32_t>(detail::popcount32(xnorResult));
                    }

                    if (remainingBits > 0)
                    {
                        const uint32_t mask = (1u << remainingBits) - 1u;
                        const uint32_t xnorResult = ~(packedInput[fullWords] ^ mPackedWeights[weightWordOffset + fullWords]) & mask;
                        matchCount += static_cast<int32_t>(detail::popcount32(xnorResult));
                    }
                }
                else
                {
                    // Unaligned case: extract bits one at a time
                    for (size_t i = 0; i < InputSize; ++i)
                    {
                        const bool inputBit = detail::getBit(packedInput, i);
                        const bool weightBit = detail::getBit(mPackedWeights, weightBitOffset + i);
                        if (inputBit == weightBit)
                        {
                            ++matchCount;
                        }
                    }
                }

                // dot = 2 * matches - InputSize (maps popcount to {-1,+1} dot product)
                const int32_t dotProduct = 2 * matchCount - static_cast<int32_t>(InputSize);
                output[j] = intToValue(dotProduct) + mBias[j];
            }
        }

        /**
         * Backward pass using the straight-through estimator (STE).
         *
         * Gradients pass through the sign() binarization unchanged,
         * but are clipped to zero for latent weights outside [-1, +1].
         *
         * @param outputDeltas Array of OutputSize gradient values from downstream
         * @param input        The same input array used in the forward pass
         * @param inputDeltas  Array of InputSize gradient values to propagate upstream (may be nullptr)
         */
        void backward(ValueType const* const outputDeltas, ValueType const* const input, ValueType* inputDeltas)
        {
            // Zero gradients
            for (size_t i = 0; i < TotalBinaryWeights; ++i)
            {
                mLatentGradients[i] = zero();
            }
            for (size_t j = 0; j < OutputSize; ++j)
            {
                mBiasGradients[j] = zero();
            }

            if (inputDeltas != nullptr)
            {
                for (size_t i = 0; i < InputSize; ++i)
                {
                    inputDeltas[i] = zero();
                }
            }

            for (size_t j = 0; j < OutputSize; ++j)
            {
                const ValueType delta = outputDeltas[j];

                // Bias gradient
                mBiasGradients[j] = delta;

                for (size_t i = 0; i < InputSize; ++i)
                {
                    // Gradient for latent weight: delta * sign(input[i])
                    // STE: pass through if |latent_weight| <= 1
                    const ValueType signInput = (input[i] < zero()) ? negOne() : one();
                    const ValueType grad = delta * signInput;

                    // STE clipping: zero gradient if latent weight outside [-1, 1]
                    const ValueType& lw = mLatentWeights[j * InputSize + i];
                    if (!(lw < negOne()) && !(one() < lw))
                    {
                        mLatentGradients[j * InputSize + i] = grad;
                    }

                    // Propagate to input: delta * sign(weight)
                    if (inputDeltas != nullptr)
                    {
                        const bool weightBit = detail::getBit(mPackedWeights, j * InputSize + i);
                        const ValueType signWeight = weightBit ? one() : negOne();
                        inputDeltas[i] += delta * signWeight;
                    }
                }
            }
        }

        /**
         * Update latent weights and biases, then re-binarize.
         * @param learningRate Step size for weight update
         */
        void updateWeights(const ValueType& learningRate)
        {
            for (size_t i = 0; i < TotalBinaryWeights; ++i)
            {
                mLatentWeights[i] += learningRate * mLatentGradients[i];
            }
            for (size_t j = 0; j < OutputSize; ++j)
            {
                mBias[j] += learningRate * mBiasGradients[j];
            }

            // Re-binarize: pack sign(latent_weight) into packed weights
            binarizeWeights();
        }

        /**
         * Set a latent weight value (for initialization or serialization).
         * Call binarizeWeights() after setting all latent weights.
         */
        void setLatentWeight(const size_t outputIdx, const size_t inputIdx, const ValueType& value)
        {
            mLatentWeights[outputIdx * InputSize + inputIdx] = value;
        }

        ValueType getLatentWeight(const size_t outputIdx, const size_t inputIdx) const
        {
            return mLatentWeights[outputIdx * InputSize + inputIdx];
        }

        void setBias(const size_t outputIdx, const ValueType& value)
        {
            mBias[outputIdx] = value;
        }

        ValueType getBias(const size_t outputIdx) const
        {
            return mBias[outputIdx];
        }

        /**
         * Get the binary weight value (+1 or -1 as ValueType).
         */
        ValueType getBinaryWeight(const size_t outputIdx, const size_t inputIdx) const
        {
            return detail::getBit(mPackedWeights, outputIdx * InputSize + inputIdx) ? one() : negOne();
        }

        /**
         * Binarize latent weights into packed binary representation.
         * Must be called after setting latent weights and before forward().
         */
        void binarizeWeights()
        {
            for (size_t i = 0; i < PackedWeightWords; ++i)
            {
                mPackedWeights[i] = 0;
            }
            for (size_t i = 0; i < TotalBinaryWeights; ++i)
            {
                detail::setBit(mPackedWeights, i, !(mLatentWeights[i] < zero()));
            }
        }

        /**
         * Get the packed weight word at the given index (for inspection/serialization).
         */
        uint32_t getPackedWeightWord(const size_t index) const
        {
            return mPackedWeights[index];
        }

        void setPackedWeightWord(const size_t index, const uint32_t value)
        {
            mPackedWeights[index] = value;
        }

    private:
        uint32_t mPackedWeights[PackedWeightWords];
        ValueType mLatentWeights[TotalBinaryWeights];
        ValueType mLatentGradients[TotalBinaryWeights];
        ValueType mBias[OutputSize];
        ValueType mBiasGradients[OutputSize];

        typedef ValueConverter<double, ValueType> Converter;

        static ValueType zero()
        {
            static const ValueType z = Converter::convertToDestinationType(0.0);
            return z;
        }

        static ValueType one()
        {
            static const ValueType o = Converter::convertToDestinationType(1.0);
            return o;
        }

        static ValueType negOne()
        {
            static const ValueType n = Converter::convertToDestinationType(-1.0);
            return n;
        }

        static ValueType intToValue(const int32_t v)
        {
            return Converter::convertToDestinationType(static_cast<double>(v));
        }

        static_assert(InputSize > 0, "Input size must be > 0.");
        static_assert(OutputSize > 0, "Output size must be > 0.");
    };
}
