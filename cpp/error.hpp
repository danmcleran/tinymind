/**
* Copyright (c) 2020 Intel Corporation
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

#include "lookupTable.hpp"
#include "log.hpp"
#include "lookupFunctions.hpp"
#include "constants.hpp"

namespace tinymind {
    template<typename ValueType, unsigned NumberOfOutputNeurons>
    struct MeanSquaredErrorCalculator
    {
        /**
        * Mean-squared error
        */
        static ValueType calculateError(ValueType const* const targetValues, ValueType const* const outputValues)
        {
            ValueType error(0);

            for (uint32_t neuron = 0; neuron < NumberOfOutputNeurons; neuron++)
            {
                const ValueType delta = (targetValues[neuron] - outputValues[neuron]);
                const ValueType deltaSquared = (delta * delta);
                error += deltaSquared;
            }

            if (NumberOfOutputNeurons > 1)
            {
                error /= NumberOfOutputNeurons;
            }

            return error;
        }
    };

    template<typename ValueType, unsigned NumberOfOutputNeurons>
    struct CrossEntropyLossCalculator
    {
        /**
         * Cross-entropy loss. log comes from LogLookup, which is accurate over
         * the whole positive range of the format; an output of 0 or below
         * takes minus the largest value of the format as its log.
         */
        static ValueType calculateError(ValueType const* const targetValues, ValueType const* const outputValues)
        {
            ValueType error(0);

            for (uint32_t neuron = 0; neuron < NumberOfOutputNeurons; neuron++)
            {
                error += (targetValues[neuron] * LogLookup<ValueType>::log(outputValues[neuron]));
            }

            error *= Constants<ValueType>::negativeOne();

            return error;
        }
    };
}
