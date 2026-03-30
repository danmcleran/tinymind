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

#include <cmath>
#include <cstddef>

namespace tinymind {
    namespace detail {
        /**
         * Fixed-point square root approximation.
         * Converts to double, computes sqrt, then converts back to QValue.
         */
        template<typename ValueType>
        struct FixedPointSqrt
        {
            typedef typename ValueType::FullWidthValueType FullWidthValueType;

            static ValueType sqrt(const ValueType& value)
            {
                static const double factor = std::pow(2.0, -1.0 * static_cast<double>(ValueType::NumberOfFractionalBits));
                const double dval = static_cast<double>(value.getValue()) * factor;
                const double result = std::sqrt(dval >= 0.0 ? dval : 0.0);
                const FullWidthValueType rawResult = static_cast<FullWidthValueType>(result * static_cast<double>(1ULL << ValueType::NumberOfFractionalBits));
                return ValueType(rawResult);
            }
        };
    }

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
     * The mDeltaWeight (first moment) storage is unused by RMSprop.
     *
     * Template parameters use the QValue(int, unsigned) constructor for
     * fixed-point types. For floating-point, use RmsPropOptimizerFloat.
     *
     * Default hyperparameters:
     *   decay = 0.9, epsilon = 1e-8
     *
     * For fixed-point Q8.8:
     *   decay ≈ 230/256 ≈ 0.898  -> (0, 230)
     *   epsilon ≈ 1/256 ≈ 0.004  -> (0, 1)
     *
     * Update rule:
     *   v = decay * v + (1 - decay) * gradient^2
     *   weight += lr * gradient / (sqrt(v) + epsilon)
     */
    template<typename ValueType,
             int DecayInt = 0, unsigned DecayFrac = 230,
             int EpsilonInt = 0, unsigned EpsilonFrac = 1>
    struct RmsPropOptimizer
    {
        // Reuse the Adam dispatch path in BackPropagationParent.
        // The AdamTag code path simply delegates to the optimizer's
        // updateWeights method, which works for any adaptive optimizer.
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

        template<typename LayerType, typename PreviousLayerType>
        void updateWeights(PreviousLayerType& previousLayer, const size_t neuron, const size_t conn, const ValueType& gradient, const ValueType& learningRate)
        {
            static const ValueType decay(DecayInt, DecayFrac);
            static const ValueType epsilon(EpsilonInt, EpsilonFrac);
            static const ValueType one(1, 0);

            // Get current second moment (running average of squared gradients)
            ValueType v = previousLayer.getSecondMomentForNeuronAndConnection(conn, neuron);

            // Update second moment: v = decay * v + (1 - decay) * gradient^2
            v = decay * v + (one - decay) * gradient * gradient;

            // Store updated moment
            previousLayer.setSecondMomentForNeuronAndConnection(conn, neuron, v);

            // Compute update: lr * gradient / (sqrt(v) + epsilon)
            const ValueType sqrtV = detail::FixedPointSqrt<ValueType>::sqrt(v);
            const ValueType update = learningRate * gradient / (sqrtV + epsilon);

            // Apply weight update
            const ValueType currentWeight = previousLayer.getWeightForNeuronAndConnection(conn, neuron);
            previousLayer.setWeightForNeuronAndConnection(conn, neuron, currentWeight + update);
        }

    private:
        size_t mTimestep;
    };

    /**
     * Floating-point RMSprop optimizer with standard hyperparameters.
     * Uses double-precision constants directly instead of QValue constructors.
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
            // Get current second moment
            ValueType v = previousLayer.getSecondMomentForNeuronAndConnection(conn, neuron);

            // Update second moment: v = decay * v + (1 - decay) * gradient^2
            v = static_cast<ValueType>(mDecay) * v + static_cast<ValueType>(1.0 - mDecay) * gradient * gradient;

            // Store updated moment
            previousLayer.setSecondMomentForNeuronAndConnection(conn, neuron, v);

            // Compute update: lr * gradient / (sqrt(v) + epsilon)
            const ValueType sqrtV = static_cast<ValueType>(std::sqrt(static_cast<double>(v)));
            const ValueType update = learningRate * gradient / (sqrtV + static_cast<ValueType>(mEpsilon));

            // Apply weight update
            const ValueType currentWeight = previousLayer.getWeightForNeuronAndConnection(conn, neuron);
            previousLayer.setWeightForNeuronAndConnection(conn, neuron, currentWeight + update);
        }

    private:
        size_t mTimestep;
        double mDecay;
        double mEpsilon;
    };

}
