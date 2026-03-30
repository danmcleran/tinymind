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
#include <cstdlib>

namespace tinymind {
    /**
     * Inverted dropout layer for regularization during training.
     *
     * During training, randomly zeros elements with probability p and
     * scales remaining elements by 1/(1-p). This "inverted" approach
     * means no scaling is needed at inference time — the forward pass
     * simply copies values through unchanged.
     *
     * Designed as a standalone composable layer (like Conv1D and Pool1D)
     * that sits between network stages in a processing pipeline:
     *
     *   Conv1D → Pool1D → Dropout → Dense network
     *
     * The dropout probability is specified as an integer percentage
     * (0-99) to avoid floating-point template parameters.
     *
     * @tparam ValueType        Numeric type (QValue or float/double)
     * @tparam Size             Number of elements in the layer
     * @tparam DropoutPercent   Probability of zeroing each element (0-99, default 50)
     */
    template<
        typename ValueType,
        size_t Size,
        unsigned DropoutPercent = 50>
    class Dropout
    {
    public:
        Dropout() : mTraining(true)
        {
            for (size_t i = 0; i < Size; ++i)
            {
                mMask[i] = true;
            }
        }

        /**
         * Forward pass with inverted dropout.
         *
         * When training: generates a new random mask, zeros dropped
         * elements, and scales survivors by 1/(1-p).
         * When not training: copies input to output unchanged.
         *
         * @param input  Array of Size values
         * @param output Array of Size values
         */
        void forward(ValueType const* const input, ValueType* output)
        {
            if (mTraining)
            {
                generateMask();

                for (size_t i = 0; i < Size; ++i)
                {
                    if (mMask[i])
                    {
                        output[i] = input[i] * scale();
                    }
                    else
                    {
                        output[i] = ValueType(0);
                    }
                }
            }
            else
            {
                for (size_t i = 0; i < Size; ++i)
                {
                    output[i] = input[i];
                }
            }
        }

        /**
         * Backward pass: apply the same mask used in the forward pass.
         * Gradients for dropped elements are zeroed; survivors are
         * scaled by the same 1/(1-p) factor.
         *
         * @param outputDeltas Array of Size gradient values from downstream
         * @param inputDeltas  Array of Size gradient values to propagate upstream
         */
        void backward(ValueType const* const outputDeltas, ValueType* inputDeltas) const
        {
            for (size_t i = 0; i < Size; ++i)
            {
                if (mMask[i])
                {
                    inputDeltas[i] = outputDeltas[i] * scale();
                }
                else
                {
                    inputDeltas[i] = ValueType(0);
                }
            }
        }

        /**
         * Set training mode. When true, dropout mask is applied.
         * When false, forward pass is identity.
         */
        void setTraining(const bool training)
        {
            mTraining = training;
        }

        bool isTraining() const
        {
            return mTraining;
        }

        /**
         * Get the current mask value for a given index.
         * True means the element is kept; false means dropped.
         */
        bool getMask(const size_t index) const
        {
            return mMask[index];
        }

    private:
        bool mMask[Size];
        bool mTraining;

        void generateMask()
        {
            for (size_t i = 0; i < Size; ++i)
            {
                const unsigned r = static_cast<unsigned>(rand() % 100);
                mMask[i] = (r >= DropoutPercent);
            }
        }

        static ValueType scale()
        {
            // 1 / (1 - p) where p = DropoutPercent / 100
            // = 100 / (100 - DropoutPercent)
            static const ValueType s = static_cast<ValueType>(100.0 / (100.0 - static_cast<double>(DropoutPercent)));
            return s;
        }

        static_assert(Size > 0, "Dropout size must be > 0.");
        static_assert(DropoutPercent < 100, "Dropout percent must be < 100.");
    };
}
