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

#include "adam.hpp"

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
#include <cmath>
#endif

namespace tinymind {
    /**
     * RMSprop optimizer policy for use with BackPropagationParent.
     *
     * RMSprop maintains a per-parameter running average of squared gradients
     * (second moment) to normalize the learning rate. Unlike Adam, it does
     * not track the first moment (mean of gradients), making it lighter
     * weight while still providing adaptive per-parameter learning rates.
     *
     * RMSprop is often preferred over Adam for recurrent networks (LSTM, GRU)
     * where the simpler update rule can provide more stable training.
     *
     * The second moment reuses the existing mPreviousDeltaWeight storage in
     * TrainableConnection, so no additional per-connection memory is required.
     * The mDeltaWeight (first moment) storage is unused by RMSprop. Bias
     * connections are updated the same way, through updateBiasWeights().
     *
     * Each constant is an (IntegerPart, FractionalPart) pair for the
     * QValue(int, fraction) constructor; FractionalPart is a raw count of
     * LSBs in ValueType's own format. The defaults are computed for
     * ValueType's format by QFractionLsbs (cpp/include/tinymind_traits.hpp):
     *   decay = 0.9, epsilon = 1e-8
     * each rounded to the format and kept inside (0, 1): in Q8.8,
     * decay = 230/256 and epsilon = 1/256; in Q16.16, decay = 58982/65536
     * and epsilon = 1/65536. For floating-point, use RmsPropOptimizerFloat.
     *
     * Update rule:
     *   v = decay * v + (1 - decay) * gradient^2
     *   weight += lr * gradient / (sqrt(v) + epsilon)
     */
    template<typename ValueType,
             int DecayInt = 0, unsigned long long DecayFrac = QFractionLsbs<ValueType, 9, 10>::value,
             int EpsilonInt = 0, unsigned long long EpsilonFrac = QFractionLsbs<ValueType, 1, 100000000>::value>
    struct RmsPropOptimizer
    {
        // Reuse the Adam dispatch path in BackPropagationParent.
        // The AdamTag code path simply delegates to the optimizer's
        // updateWeights/updateBiasWeights methods, which works for any
        // adaptive optimizer.
        static const bool IsAdam = true;

        RmsPropOptimizer() : mTimestep(0)
        {
        }

        void initialize()
        {
            mTimestep = 0;
        }

        void step()
        {
            ++mTimestep;
        }

        size_t getTimestep() const
        {
            return mTimestep;
        }

        static ValueType decay()
        {
            static const ValueType value(static_cast<typename ValueType::FixedPartFieldType>(DecayInt), static_cast<typename ValueType::FractionalPartFieldType>(DecayFrac));
            return value;
        }

        static ValueType epsilon()
        {
            static const ValueType value(static_cast<typename ValueType::FixedPartFieldType>(EpsilonInt), static_cast<typename ValueType::FractionalPartFieldType>(EpsilonFrac));
            return value;
        }

        template<typename LayerType, typename PreviousLayerType>
        void updateWeights(PreviousLayerType& previousLayer, const size_t neuron, const size_t conn, const ValueType& gradient, const ValueType& learningRate)
        {
            ValueType v = previousLayer.getSecondMomentForNeuronAndConnection(neuron, conn);
            const ValueType update = computeUpdate(v, gradient, learningRate);
            previousLayer.setSecondMomentForNeuronAndConnection(neuron, conn, v);

            const ValueType currentWeight = previousLayer.getWeightForNeuronAndConnection(neuron, conn);
            previousLayer.setWeightForNeuronAndConnection(neuron, conn, currentWeight + update);
        }

        template<typename LayerType, typename PreviousLayerType>
        void updateBiasWeights(PreviousLayerType& previousLayer, const size_t conn, const ValueType& gradient, const ValueType& learningRate)
        {
            ValueType v = previousLayer.getBiasNeuronSecondMomentForConnection(conn);
            const ValueType update = computeUpdate(v, gradient, learningRate);
            previousLayer.setBiasNeuronSecondMomentForConnection(conn, v);

            const ValueType currentWeight = previousLayer.getBiasNeuronWeightForConnection(conn);
            previousLayer.setBiasNeuronWeightForConnection(conn, currentWeight + update);
        }

    private:
        // Updates v in place and returns lr * gradient / (sqrt(v) + epsilon).
        static ValueType computeUpdate(ValueType& v, const ValueType& gradient, const ValueType& learningRate)
        {
            const ValueType one(static_cast<typename ValueType::FixedPartFieldType>(1), static_cast<typename ValueType::FractionalPartFieldType>(0));

            v = decay() * v + (one - decay()) * gradient * gradient;

            const ValueType sqrtV = SquareRootApproximation<ValueType>::sqrt(v);
            return learningRate * gradient / (sqrtV + epsilon());
        }

        size_t mTimestep;
    };

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
    /**
     * Floating-point RMSprop optimizer with standard hyperparameters
     * (decay = 0.9, epsilon = 1e-8). Uses double-precision constants
     * directly instead of QValue constructors. Requires TINYMIND_ENABLE_STD
     * because std::sqrt is needed for the second-moment normalization.
     */
    template<typename ValueType>
    struct RmsPropOptimizerFloat
    {
        static const bool IsAdam = true;

        RmsPropOptimizerFloat() : mTimestep(0), mDecay(0.9), mEpsilon(1e-8)
        {
        }

        void initialize()
        {
            mTimestep = 0;
        }

        void step()
        {
            ++mTimestep;
        }

        size_t getTimestep() const
        {
            return mTimestep;
        }

        template<typename LayerType, typename PreviousLayerType>
        void updateWeights(PreviousLayerType& previousLayer, const size_t neuron, const size_t conn, const ValueType& gradient, const ValueType& learningRate)
        {
            ValueType v = previousLayer.getSecondMomentForNeuronAndConnection(neuron, conn);
            const ValueType update = this->computeUpdate(v, gradient, learningRate);
            previousLayer.setSecondMomentForNeuronAndConnection(neuron, conn, v);

            const ValueType currentWeight = previousLayer.getWeightForNeuronAndConnection(neuron, conn);
            previousLayer.setWeightForNeuronAndConnection(neuron, conn, currentWeight + update);
        }

        template<typename LayerType, typename PreviousLayerType>
        void updateBiasWeights(PreviousLayerType& previousLayer, const size_t conn, const ValueType& gradient, const ValueType& learningRate)
        {
            ValueType v = previousLayer.getBiasNeuronSecondMomentForConnection(conn);
            const ValueType update = this->computeUpdate(v, gradient, learningRate);
            previousLayer.setBiasNeuronSecondMomentForConnection(conn, v);

            const ValueType currentWeight = previousLayer.getBiasNeuronWeightForConnection(conn);
            previousLayer.setBiasNeuronWeightForConnection(conn, currentWeight + update);
        }

    private:
        ValueType computeUpdate(ValueType& v, const ValueType& gradient, const ValueType& learningRate) const
        {
            v = static_cast<ValueType>(mDecay) * v + static_cast<ValueType>(1.0 - mDecay) * gradient * gradient;

            const ValueType sqrtV = static_cast<ValueType>(std::sqrt(static_cast<double>(v)));
            return learningRate * gradient / (sqrtV + static_cast<ValueType>(mEpsilon));
        }

        size_t mTimestep;
        double mDecay;
        double mEpsilon;
    };
#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD

}
