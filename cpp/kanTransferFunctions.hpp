/**
* Copyright (c) 2025 Dan McLeran
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

#include "activationFunctions.hpp"
#include "error.hpp"
#include "nninit.hpp"
#include "zeroTolerance.hpp"
#include "constants.hpp"

namespace tinymind {

    /**
     * SiLU (Sigmoid Linear Unit) activation policy.
     *
     * SiLU(x) = x * sigmoid(x)
     *
     * Used as the residual activation in KAN edge functions.
     * Reuses existing SigmoidActivationPolicy lookup tables for fixed-point.
     */
    template<typename ValueType>
    struct SiLUActivationPolicy
    {
        static ValueType activationFunction(const ValueType& value)
        {
            const ValueType sig = SigmoidActivationPolicy<ValueType>::activationFunction(value);
            return value * sig;
        }

        /**
         * SiLU derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
         *                = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
         */
        static ValueType activationFunctionDerivative(const ValueType& value)
        {
            const ValueType sig = SigmoidActivationPolicy<ValueType>::activationFunction(value);
            const ValueType oneMinusSig = Constants<ValueType>::one() - sig;
            return sig * (Constants<ValueType>::one() + value * oneMinusSig);
        }
    };

    /**
     * KAN-specific transfer functions policy.
     *
     * Similar to FixedPointTransferFunctions but without hidden/output neuron
     * activation functions, since KAN nodes are pure summation nodes.
     * The learnable activation functions live on the edges (connections).
     */
    template<
            typename ValueType,
            class KanRandomNumberGeneratorPolicy,
            unsigned NumberOfOutputNeurons = 1,
            class KanNetworkInitializationPolicy = tinymind::DefaultNetworkInitializer<ValueType>,
            class KanErrorCalculatorPolicy = tinymind::MeanSquaredErrorCalculator<ValueType, NumberOfOutputNeurons>,
            class KanZeroTolerancePolicy = tinymind::ZeroToleranceCalculator<ValueType> >
    struct KanTransferFunctions
    {
        typedef ValueType TransferFunctionsValueType;
        typedef KanRandomNumberGeneratorPolicy RandomNumberGeneratorPolicy;
        typedef KanNetworkInitializationPolicy NetworkInitializationPolicy;
        typedef KanErrorCalculatorPolicy ErrorCalculatorPolicy;
        typedef KanZeroTolerancePolicy ZeroToleranceCalculatorPolicy;

        static const unsigned NumberOfTransferFunctionsOutputNeurons = NumberOfOutputNeurons;

        static ValueType calculateError(ValueType const* const targetValues, ValueType const* const outputValues)
        {
            return ErrorCalculatorPolicy::calculateError(targetValues, outputValues);
        }

        static ValueType generateRandomWeight()
        {
            return RandomNumberGeneratorPolicy::generateRandomWeight();
        }

        static ValueType initialAccelerationRate()
        {
            return NetworkInitializationPolicy::initialAccelerationRate();
        }

        static ValueType initialBiasOutputValue()
        {
            return NetworkInitializationPolicy::initialBiasOutputValue();
        }

        static ValueType initialDeltaWeight()
        {
            return NetworkInitializationPolicy::initialDeltaWeight();
        }

        static ValueType initialGradientValue()
        {
            return NetworkInitializationPolicy::initialGradientValue();
        }

        static ValueType initialLearningRate()
        {
            return NetworkInitializationPolicy::initialLearningRate();
        }

        static ValueType initialMomentumRate()
        {
            return NetworkInitializationPolicy::initialMomentumRate();
        }

        static ValueType initialOutputValue()
        {
            return NetworkInitializationPolicy::initialOutputValue();
        }

        static bool isWithinZeroTolerance(const ValueType& value)
        {
            return ZeroToleranceCalculatorPolicy::isWithinZeroTolerance(value);
        }

        static ValueType negate(const ValueType& value)
        {
            return value * Constants<ValueType>::negativeOne();
        }

        static ValueType noOpDeltaWeight()
        {
            return NetworkInitializationPolicy::noOpDeltaWeight();
        }

        static ValueType noOpWeight()
        {
            return NetworkInitializationPolicy::noOpWeight();
        }
    };
}
