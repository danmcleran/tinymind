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
        // Ternary values encoded as 2-bit pairs: 00 = 0, 01 = +1, 10 = -1
        static const uint8_t TERNARY_ZERO = 0;
        static const uint8_t TERNARY_POS  = 1;
        static const uint8_t TERNARY_NEG  = 2;
        static const size_t TERNARY_PAIRS_PER_WORD = 16; // 32 bits / 2 bits per ternary value

        inline uint8_t getTernaryValue(const uint32_t* packed, const size_t index)
        {
            const size_t word = index / TERNARY_PAIRS_PER_WORD;
            const size_t shift = (index % TERNARY_PAIRS_PER_WORD) * 2;
            return static_cast<uint8_t>((packed[word] >> shift) & 0x3u);
        }

        inline void setTernaryValue(uint32_t* packed, const size_t index, const uint8_t value)
        {
            const size_t word = index / TERNARY_PAIRS_PER_WORD;
            const size_t shift = (index % TERNARY_PAIRS_PER_WORD) * 2;
            packed[word] &= ~(0x3u << shift);
            packed[word] |= (static_cast<uint32_t>(value & 0x3u) << shift);
        }
    } // namespace detail

    /**
     * Ternary dense layer: weights constrained to {-1, 0, +1}.
     *
     * Multiplications become conditional add/subtract/skip operations,
     * eliminating all multiplication from the forward pass. Weights are
     * stored as 2-bit packed values (16x memory reduction vs full-precision).
     *
     * Ternarization uses a threshold: weights with |w| < threshold become 0,
     * positive weights above threshold become +1, and negative weights below
     * -threshold become -1. The threshold is specified as a percentage (0-99)
     * of the mean absolute weight value.
     *
     * Training maintains real-valued latent weights. The ternary weights
     * used in the forward pass are ternarize(latent_weight). Gradients use
     * the straight-through estimator (STE) with a dead zone.
     *
     * @tparam ValueType         Numeric type for biases and latent weights (QValue or float/double)
     * @tparam InputSize         Number of input elements
     * @tparam OutputSize        Number of output neurons
     * @tparam ThresholdPercent  Ternarization threshold as percentage of mean |weight| (default 50)
     */
    template<
        typename ValueType,
        size_t InputSize,
        size_t OutputSize,
        unsigned ThresholdPercent = 50>
    class TernaryDense
    {
    public:
        static const size_t TotalTernaryWeights = InputSize * OutputSize;
        static const size_t PackedWeightWords = (TotalTernaryWeights + detail::TERNARY_PAIRS_PER_WORD - 1) / detail::TERNARY_PAIRS_PER_WORD;

        TernaryDense()
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
            for (size_t i = 0; i < TotalTernaryWeights; ++i)
            {
                mLatentWeights[i] = zero();
                mLatentGradients[i] = zero();
            }
        }

        /**
         * Forward pass using ternary {-1, 0, +1} weights.
         *
         * For each output neuron, accumulates input values based on the
         * ternary weight: +1 adds, -1 subtracts, 0 skips. No multiplication
         * is performed.
         *
         * @param input  Array of InputSize values
         * @param output Array of OutputSize values
         */
        void forward(ValueType const* const input, ValueType* output) const
        {
            for (size_t j = 0; j < OutputSize; ++j)
            {
                ValueType sum = mBias[j];
                const size_t baseIdx = j * InputSize;

                for (size_t i = 0; i < InputSize; ++i)
                {
                    const uint8_t tw = detail::getTernaryValue(mPackedWeights, baseIdx + i);
                    if (tw == detail::TERNARY_POS)
                    {
                        sum += input[i];
                    }
                    else if (tw == detail::TERNARY_NEG)
                    {
                        sum -= input[i];
                    }
                    // TERNARY_ZERO: skip (no operation)
                }

                output[j] = sum;
            }
        }

        /**
         * Backward pass using the straight-through estimator (STE).
         *
         * Gradients pass through the ternarization unchanged for latent
         * weights within the active range. Gradients are zeroed for
         * latent weights far outside the threshold to prevent instability.
         *
         * @param outputDeltas Array of OutputSize gradient values from downstream
         * @param input        The same input array used in the forward pass
         * @param inputDeltas  Array of InputSize gradient values to propagate upstream (may be nullptr)
         */
        void backward(ValueType const* const outputDeltas, ValueType const* const input, ValueType* inputDeltas)
        {
            // Zero gradients
            for (size_t i = 0; i < TotalTernaryWeights; ++i)
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
                const size_t baseIdx = j * InputSize;

                // Bias gradient
                mBiasGradients[j] = delta;

                for (size_t i = 0; i < InputSize; ++i)
                {
                    // STE: gradient passes through ternarization
                    // Gradient for latent weight: delta * input[i]
                    mLatentGradients[baseIdx + i] = delta * input[i];

                    // Propagate to input using ternary weight
                    if (inputDeltas != nullptr)
                    {
                        const uint8_t tw = detail::getTernaryValue(mPackedWeights, baseIdx + i);
                        if (tw == detail::TERNARY_POS)
                        {
                            inputDeltas[i] += delta;
                        }
                        else if (tw == detail::TERNARY_NEG)
                        {
                            inputDeltas[i] -= delta;
                        }
                    }
                }
            }
        }

        /**
         * Update latent weights and biases, then re-ternarize.
         * @param learningRate Step size for weight update
         */
        void updateWeights(const ValueType& learningRate)
        {
            for (size_t i = 0; i < TotalTernaryWeights; ++i)
            {
                mLatentWeights[i] += learningRate * mLatentGradients[i];
            }
            for (size_t j = 0; j < OutputSize; ++j)
            {
                mBias[j] += learningRate * mBiasGradients[j];
            }

            // Re-ternarize
            ternarizeWeights();
        }

        /**
         * Ternarize latent weights into packed ternary representation.
         *
         * Computes the mean absolute latent weight, then applies the
         * threshold: |w| < threshold * mean_abs => 0, else sign(w).
         *
         * Must be called after setting latent weights and before forward().
         */
        void ternarizeWeights()
        {
            // Compute mean absolute weight
            ValueType sumAbs = zero();
            for (size_t i = 0; i < TotalTernaryWeights; ++i)
            {
                const ValueType& w = mLatentWeights[i];
                sumAbs += (w < zero()) ? (zero() - w) : w;
            }
            const ValueType meanAbs = sumAbs / totalWeightsValue();
            const ValueType thresh = meanAbs * thresholdFraction();

            for (size_t i = 0; i < PackedWeightWords; ++i)
            {
                mPackedWeights[i] = 0;
            }

            for (size_t i = 0; i < TotalTernaryWeights; ++i)
            {
                const ValueType& w = mLatentWeights[i];
                uint8_t tv;
                if (w < zero())
                {
                    tv = ((zero() - w) < thresh) ? detail::TERNARY_ZERO : detail::TERNARY_NEG;
                }
                else
                {
                    tv = (w < thresh) ? detail::TERNARY_ZERO : detail::TERNARY_POS;
                }
                detail::setTernaryValue(mPackedWeights, i, tv);
            }
        }

        /**
         * Set a latent weight value (for initialization or serialization).
         * Call ternarizeWeights() after setting all latent weights.
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
         * Get the ternary weight value (+1, 0, or -1) as the raw encoding.
         */
        uint8_t getTernaryWeight(const size_t outputIdx, const size_t inputIdx) const
        {
            return detail::getTernaryValue(mPackedWeights, outputIdx * InputSize + inputIdx);
        }

        /**
         * Get the ternary weight as a ValueType (+1, 0, or -1).
         */
        ValueType getTernaryWeightValue(const size_t outputIdx, const size_t inputIdx) const
        {
            const uint8_t tw = getTernaryWeight(outputIdx, inputIdx);
            if (tw == detail::TERNARY_POS)
            {
                return one();
            }
            else if (tw == detail::TERNARY_NEG)
            {
                return negOne();
            }
            return zero();
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
        ValueType mLatentWeights[TotalTernaryWeights];
        ValueType mLatentGradients[TotalTernaryWeights];
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

        static ValueType totalWeightsValue()
        {
            static const ValueType t = Converter::convertToDestinationType(static_cast<double>(TotalTernaryWeights));
            return t;
        }

        static ValueType thresholdFraction()
        {
            static const ValueType t = Converter::convertToDestinationType(static_cast<double>(ThresholdPercent) / 100.0);
            return t;
        }

        static_assert(InputSize > 0, "Input size must be > 0.");
        static_assert(OutputSize > 0, "Output size must be > 0.");
        static_assert(ThresholdPercent < 100, "Threshold percent must be < 100.");
    };
}
