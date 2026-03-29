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

#include <cmath>
#include <cstddef>

namespace tinymind {
    /**
     * No-op optimizer policy (default). Weight updates use SGD with momentum.
     */
    template<typename ValueType>
    struct NullOptimizerPolicy
    {
        static const bool IsAdam = false;

        void initialize() {}
        void step() {}
    };

    /**
     * Square root approximation for floating-point types.
     */
    template<typename ValueType>
    struct SquareRootApproximation
    {
        static ValueType sqrt(const ValueType& value)
        {
            return static_cast<ValueType>(std::sqrt(static_cast<double>(value)));
        }
    };

    /**
     * Adam optimizer policy for use with BackPropagationParent.
     *
     * Adam maintains per-parameter first-moment (mean) and second-moment
     * (uncentered variance) estimates, enabling adaptive learning rates.
     * This is especially valuable for gated networks (LSTM, GRU) where
     * different gates need different effective learning rates.
     *
     * The first and second moments reuse the existing mDeltaWeight and
     * mPreviousDeltaWeight storage in TrainableConnection, so no additional
     * per-connection memory is required beyond what SGD already uses.
     *
     * Template parameters use the QValue(int, unsigned) constructor for
     * fixed-point types. For floating-point, provide your own specialization
     * or use AdamOptimizerFloat.
     *
     * Default hyperparameters:
     *   beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8
     *
     * For fixed-point Q8.8:
     *   beta1 ≈ 230/256 ≈ 0.898  -> (0, 230)
     *   beta2 ≈ 255/256 ≈ 0.996  -> (0, 255)
     *   epsilon ≈ 1/256 ≈ 0.004  -> (0, 1)
     */
    template<typename ValueType,
             int Beta1Int = 0, unsigned Beta1Frac = 230,
             int Beta2Int = 0, unsigned Beta2Frac = 255,
             int EpsilonInt = 0, unsigned EpsilonFrac = 1>
    struct AdamOptimizer
    {
        static const bool IsAdam = true;

        AdamOptimizer() : mTimestep(0)
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
            static const ValueType beta1(Beta1Int, Beta1Frac);
            static const ValueType beta2(Beta2Int, Beta2Frac);
            static const ValueType epsilon(EpsilonInt, EpsilonFrac);
            static const ValueType one(1, 0);

            // Get current moments
            ValueType m = previousLayer.getFirstMomentForNeuronAndConnection(conn, neuron);
            ValueType v = previousLayer.getSecondMomentForNeuronAndConnection(conn, neuron);

            // Update biased first moment: m = beta1 * m + (1 - beta1) * gradient
            m = beta1 * m + (one - beta1) * gradient;

            // Update biased second moment: v = beta2 * v + (1 - beta2) * gradient^2
            v = beta2 * v + (one - beta2) * gradient * gradient;

            // Store updated moments
            previousLayer.setFirstMomentForNeuronAndConnection(conn, neuron, m);
            previousLayer.setSecondMomentForNeuronAndConnection(conn, neuron, v);

            // Bias correction
            ValueType mHat = m;
            ValueType vHat = v;
            if (mTimestep < 100)
            {
                // Apply bias correction only in early timesteps when it matters
                ValueType beta1Power = one;
                ValueType beta2Power = one;
                for (size_t t = 0; t < mTimestep; ++t)
                {
                    beta1Power = beta1Power * beta1;
                    beta2Power = beta2Power * beta2;
                }
                const ValueType beta1Correction = one - beta1Power;
                const ValueType beta2Correction = one - beta2Power;

                if (beta1Correction > epsilon)
                {
                    mHat = m / beta1Correction;
                }
                if (beta2Correction > epsilon)
                {
                    vHat = v / beta2Correction;
                }
            }

            // Compute update: lr * mHat / (sqrt(vHat) + epsilon)
            const ValueType sqrtV = SquareRootApproximation<ValueType>::sqrt(vHat);
            const ValueType update = learningRate * mHat / (sqrtV + epsilon);

            // Apply weight update
            const ValueType currentWeight = previousLayer.getWeightForNeuronAndConnection(conn, neuron);
            previousLayer.setWeightForNeuronAndConnection(conn, neuron, currentWeight + update);
        }

    private:
        size_t mTimestep;
    };

    /**
     * Floating-point Adam optimizer with standard hyperparameters.
     * Uses double-precision constants directly instead of QValue constructors.
     */
    template<typename ValueType>
    struct AdamOptimizerFloat
    {
        static const bool IsAdam = true;

        AdamOptimizerFloat() : mTimestep(0), mBeta1(0.9), mBeta2(0.999), mEpsilon(1e-8)
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
            // Get current moments
            ValueType m = previousLayer.getFirstMomentForNeuronAndConnection(conn, neuron);
            ValueType v = previousLayer.getSecondMomentForNeuronAndConnection(conn, neuron);

            // Update biased moments
            m = static_cast<ValueType>(mBeta1) * m + static_cast<ValueType>(1.0 - mBeta1) * gradient;
            v = static_cast<ValueType>(mBeta2) * v + static_cast<ValueType>(1.0 - mBeta2) * gradient * gradient;

            // Store updated moments
            previousLayer.setFirstMomentForNeuronAndConnection(conn, neuron, m);
            previousLayer.setSecondMomentForNeuronAndConnection(conn, neuron, v);

            // Bias correction
            const double beta1Power = std::pow(mBeta1, static_cast<double>(mTimestep));
            const double beta2Power = std::pow(mBeta2, static_cast<double>(mTimestep));
            const ValueType mHat = m / static_cast<ValueType>(1.0 - beta1Power);
            const ValueType vHat = v / static_cast<ValueType>(1.0 - beta2Power);

            // Compute update
            const ValueType sqrtV = static_cast<ValueType>(std::sqrt(static_cast<double>(vHat)));
            const ValueType update = learningRate * mHat / (sqrtV + static_cast<ValueType>(mEpsilon));

            // Apply weight update
            const ValueType currentWeight = previousLayer.getWeightForNeuronAndConnection(conn, neuron);
            previousLayer.setWeightForNeuronAndConnection(conn, neuron, currentWeight + update);
        }

    private:
        size_t mTimestep;
        double mBeta1;
        double mBeta2;
        double mEpsilon;
    };

}
