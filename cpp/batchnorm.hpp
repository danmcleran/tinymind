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
#include <cmath>

#include "adam.hpp"
#include "nnproperties.hpp"

namespace tinymind {
    /**
     * 1D batch normalization layer for stabilizing training.
     *
     * Normalizes each element across a mini-batch to zero mean and unit
     * variance, then applies learnable affine parameters (gamma, beta):
     *
     *   y = gamma * (x - mean) / sqrt(variance + epsilon) + beta
     *
     * During training, computes per-element mean and variance from the
     * current input and maintains exponential moving averages for inference.
     * During inference, uses the stored running statistics.
     *
     * Designed as a standalone composable layer (like Conv1D, Pool1D,
     * and Dropout) that sits between network stages:
     *
     *   Conv1D -> BatchNorm1D -> ReLU -> Pool1D -> Dense network
     *
     * For single-sample training (common on embedded), the "batch" is a
     * single vector. The running mean/variance still accumulate useful
     * statistics over successive forward passes, providing smoothed
     * normalization at inference time.
     *
     * The momentum parameter controls the running average update rate
     * and is specified as an integer percentage (1-99) to avoid
     * floating-point template parameters.
     *
     * @tparam ValueType        Numeric type (QValue or float/double)
     * @tparam Size             Number of elements (features) to normalize
     * @tparam MomentumPercent  Running average momentum as percentage (default 10,
     *                          meaning running_mean = 0.9 * running_mean + 0.1 * batch_mean)
     * @tparam EpsilonPercent   Epsilon for numerical stability as percentage of 1.0
     *                          (default 1, meaning epsilon = 0.01). For fixed-point,
     *                          use a value large enough to represent in the Q format.
     */
    template<
        typename ValueType,
        size_t Size,
        unsigned MomentumPercent = 10,
        unsigned EpsilonPercent = 1>
    class BatchNorm1D
    {
    public:
        BatchNorm1D() : mTraining(true)
        {
            for (size_t i = 0; i < Size; ++i)
            {
                mGamma[i] = one();
                mBeta[i] = zero();
                mRunningMean[i] = zero();
                mRunningVariance[i] = one();
                mGammaGradient[i] = zero();
                mBetaGradient[i] = zero();
                mNormalized[i] = zero();
            }
        }

        /**
         * Forward pass: normalize input, apply affine transform.
         *
         * Training mode: computes mean/variance from input, updates
         * running statistics, normalizes, and applies gamma/beta.
         *
         * Inference mode: uses stored running mean/variance.
         *
         * @param input  Array of Size values
         * @param output Array of Size values
         */
        void forward(ValueType const* const input, ValueType* output)
        {
            if (mTraining)
            {
                // Compute batch mean
                ValueType mean = zero();
                for (size_t i = 0; i < Size; ++i)
                {
                    mean += input[i];
                }
                mean = mean / sizeValue();

                // Compute batch variance
                ValueType variance = zero();
                for (size_t i = 0; i < Size; ++i)
                {
                    const ValueType diff = input[i] - mean;
                    variance += diff * diff;
                }
                variance = variance / sizeValue();

                // Update running statistics with exponential moving average
                const ValueType alpha = momentum();
                const ValueType oneMinusAlpha = one() - alpha;
                for (size_t i = 0; i < Size; ++i)
                {
                    mRunningMean[i] = oneMinusAlpha * mRunningMean[i] + alpha * mean;
                    mRunningVariance[i] = oneMinusAlpha * mRunningVariance[i] + alpha * variance;
                }

                // Normalize and apply affine transform
                const ValueType invStd = one() / SquareRootApproximation<ValueType>::sqrt(variance + epsilon());
                for (size_t i = 0; i < Size; ++i)
                {
                    mNormalized[i] = (input[i] - mean) * invStd;
                    output[i] = mGamma[i] * mNormalized[i] + mBeta[i];
                }
            }
            else
            {
                // Inference: use running statistics
                for (size_t i = 0; i < Size; ++i)
                {
                    const ValueType invStd = one() / SquareRootApproximation<ValueType>::sqrt(mRunningVariance[i] + epsilon());
                    mNormalized[i] = (input[i] - mRunningMean[i]) * invStd;
                    output[i] = mGamma[i] * mNormalized[i] + mBeta[i];
                }
            }
        }

        /**
         * Backward pass: compute gradients for gamma, beta, and input.
         *
         * Accumulates gradients for the learnable parameters (gamma, beta)
         * and propagates normalized gradients to the input.
         *
         * @param outputDeltas Array of Size gradient values from downstream
         * @param inputDeltas  Array of Size gradient values to propagate upstream
         */
        void backward(ValueType const* const outputDeltas, ValueType* inputDeltas)
        {
            for (size_t i = 0; i < Size; ++i)
            {
                // Gradient for gamma: dL/dgamma = dL/dy * x_normalized
                mGammaGradient[i] += outputDeltas[i] * mNormalized[i];

                // Gradient for beta: dL/dbeta = dL/dy
                mBetaGradient[i] += outputDeltas[i];

                // Gradient for input: dL/dx_norm = dL/dy * gamma
                inputDeltas[i] = outputDeltas[i] * mGamma[i];
            }
        }

        /**
         * Update learnable parameters (gamma, beta) using SGD.
         *
         * @param learningRate Step size for parameter update
         */
        void updateParameters(const ValueType& learningRate)
        {
            for (size_t i = 0; i < Size; ++i)
            {
                mGamma[i] += learningRate * mGammaGradient[i];
                mBeta[i] += learningRate * mBetaGradient[i];

                mGammaGradient[i] = zero();
                mBetaGradient[i] = zero();
            }
        }

        void setTraining(const bool training)
        {
            mTraining = training;
        }

        bool isTraining() const
        {
            return mTraining;
        }

        ValueType getGamma(const size_t index) const { return mGamma[index]; }
        void setGamma(const size_t index, const ValueType& value) { mGamma[index] = value; }

        ValueType getBeta(const size_t index) const { return mBeta[index]; }
        void setBeta(const size_t index, const ValueType& value) { mBeta[index] = value; }

        ValueType getRunningMean(const size_t index) const { return mRunningMean[index]; }
        ValueType getRunningVariance(const size_t index) const { return mRunningVariance[index]; }

        ValueType getGammaGradient(const size_t index) const { return mGammaGradient[index]; }
        ValueType getBetaGradient(const size_t index) const { return mBetaGradient[index]; }

        ValueType getNormalized(const size_t index) const { return mNormalized[index]; }

    private:
        ValueType mGamma[Size];
        ValueType mBeta[Size];
        ValueType mRunningMean[Size];
        ValueType mRunningVariance[Size];
        ValueType mGammaGradient[Size];
        ValueType mBetaGradient[Size];
        ValueType mNormalized[Size];
        bool mTraining;

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

        static ValueType sizeValue()
        {
            static const ValueType s = Converter::convertToDestinationType(static_cast<double>(Size));
            return s;
        }

        static ValueType momentum()
        {
            static const ValueType m = Converter::convertToDestinationType(static_cast<double>(MomentumPercent) / 100.0);
            return m;
        }

        static ValueType epsilon()
        {
            static const ValueType e = Converter::convertToDestinationType(static_cast<double>(EpsilonPercent) / 100.0);
            return e;
        }

        static_assert(Size > 0, "BatchNorm size must be > 0.");
        static_assert(MomentumPercent > 0 && MomentumPercent <= 100, "Momentum percent must be in (0, 100].");
        static_assert(EpsilonPercent > 0, "Epsilon percent must be > 0.");
    };
}
