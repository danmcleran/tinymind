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

#include "include/tinymind_traits.hpp"

#include <cstddef>

#include "adam.hpp"
#include "nnproperties.hpp"

namespace tinymind {
    /**
     * 1D batch normalization: each of the Size features is normalized with
     * its own mean and variance, then scaled and shifted by learnable
     * per-feature gamma and beta:
     *
     *   y[n][i] = gamma[i] * (x[n][i] - mean[i]) / sqrt(var[i] + epsilon) + beta[i]
     *
     * Two ways to supply the statistics:
     *
     * Mini-batch (forwardBatch / backwardBatch, 2 <= N <= MaxBatchSize).
     *   In training mode, mean[i] and var[i] (biased) are taken over the N
     *   samples of the batch, which is standard batch normalization. The
     *   running statistics are updated per feature with an exponential
     *   moving average (the variance with the unbiased N/(N-1) estimate),
     *   and backwardBatch() propagates the exact input gradient, including
     *   the terms through the batch mean and variance. In inference mode the
     *   running statistics are used.
     *
     * Single sample (forward / backward, or forwardBatch with N = 1).
     *   One sample has no batch statistics (its per-feature variance is 0),
     *   so this path normalizes with the running statistics in both modes:
     *   training and inference compute the same function. In training mode
     *   the sample then updates the running statistics (an exponentially
     *   weighted mean and variance, Finch 2009), so the normalization tracks
     *   the input distribution as samples stream in. The statistics in use
     *   for a forward pass are those before its own update, and the gradient
     *   treats them as constants: dL/dx = dL/dy * gamma / sqrt(var + epsilon).
     *   This is streaming normalization with learned affine parameters, not
     *   mini-batch batch normalization; it starts from mean 0 and variance 1.
     *
     * Gradient convention: like the rest of tinymind, deltas passed in are
     * -dL/dy, the deltas passed back are -dL/dx, the accumulated gamma/beta
     * gradients are -dL/dgamma and -dL/dbeta, and updateParameters() adds
     * learningRate * gradient.
     *
     * The momentum parameter controls the running average update rate
     * and is specified as an integer percentage to avoid floating-point
     * template parameters.
     *
     * @tparam ValueType        Numeric type (QValue or float/double)
     * @tparam Size             Number of features to normalize
     * @tparam MomentumPercent  Running average momentum as percentage (default 10,
     *                          meaning running_mean = 0.9 * running_mean + 0.1 * batch_mean)
     * @tparam EpsilonPercent   Epsilon for numerical stability as percentage of 1.0
     *                          (default 1, meaning epsilon = 0.01). For fixed-point,
     *                          use a value large enough to represent in the Q format.
     * @tparam MaxBatchSize     Largest N accepted by forwardBatch/backwardBatch
     *                          (default 1: single-sample streaming only). The
     *                          layer stores N * Size normalized values for backward.
     */
    template<
        typename ValueType,
        size_t Size,
        unsigned MomentumPercent = 10,
        unsigned EpsilonPercent = 1,
        size_t MaxBatchSize = 1>
    class BatchNorm1D
    {
    public:
        BatchNorm1D() : mTraining(true), mUsedBatchStatistics(false), mLastBatchSize(0)
        {
            for (size_t i = 0; i < Size; ++i)
            {
                mGamma[i] = one();
                mBeta[i] = zero();
                mRunningMean[i] = zero();
                mRunningVariance[i] = one();
                mGammaGradient[i] = zero();
                mBetaGradient[i] = zero();
                mInvStd[i] = zero();
            }
            for (size_t i = 0; i < (MaxBatchSize * Size); ++i)
            {
                mNormalized[i] = zero();
            }
        }

        /**
         * Single-sample forward pass, normalized with the running statistics
         * (see the class comment). In training mode the sample then updates
         * the running statistics.
         *
         * @param input  Array of Size values
         * @param output Array of Size values
         */
        void forward(ValueType const* const input, ValueType* output)
        {
            normalizeWithRunningStatistics(input, output, 1);
            mUsedBatchStatistics = false;
            mLastBatchSize = 1;

            if (mTraining)
            {
                const ValueType alpha = momentum();
                const ValueType oneMinusAlpha = one() - alpha;
                for (size_t i = 0; i < Size; ++i)
                {
                    const ValueType diff = input[i] - mRunningMean[i];
                    mRunningMean[i] = mRunningMean[i] + alpha * diff;
                    mRunningVariance[i] = oneMinusAlpha * (mRunningVariance[i] + alpha * diff * diff);
                }
            }
        }

        /**
         * Mini-batch forward pass over batchSize samples laid out
         * input[n * Size + i]. Training mode with batchSize >= 2 uses batch
         * statistics; otherwise the running statistics (batchSize == 1 is the
         * same as forward()). A batchSize of 0 or above MaxBatchSize is
         * ignored (nothing is written).
         */
        void forwardBatch(ValueType const* const input, ValueType* output, const size_t batchSize)
        {
            if ((batchSize == 0) || (batchSize > MaxBatchSize))
            {
                return;
            }

            if (batchSize == 1)
            {
                forward(input, output);
                return;
            }

            if (!mTraining)
            {
                normalizeWithRunningStatistics(input, output, batchSize);
                mUsedBatchStatistics = false;
                mLastBatchSize = batchSize;
                return;
            }

            const ValueType n = fromInteger(static_cast<int>(batchSize));
            const ValueType nMinusOne = fromInteger(static_cast<int>(batchSize) - 1);
            const ValueType alpha = momentum();
            const ValueType oneMinusAlpha = one() - alpha;

            for (size_t i = 0; i < Size; ++i)
            {
                ValueType mean = zero();
                for (size_t sample = 0; sample < batchSize; ++sample)
                {
                    mean += input[sample * Size + i];
                }
                mean = mean / n;

                ValueType variance = zero();
                for (size_t sample = 0; sample < batchSize; ++sample)
                {
                    const ValueType diff = input[sample * Size + i] - mean;
                    variance += diff * diff;
                }
                const ValueType sumOfSquares = variance;
                variance = variance / n;

                const ValueType invStd = one() / SquareRootApproximation<ValueType>::sqrt(variance + epsilon());
                mInvStd[i] = invStd;

                for (size_t sample = 0; sample < batchSize; ++sample)
                {
                    const size_t index = sample * Size + i;
                    mNormalized[index] = (input[index] - mean) * invStd;
                    output[index] = mGamma[i] * mNormalized[index] + mBeta[i];
                }

                mRunningMean[i] = oneMinusAlpha * mRunningMean[i] + alpha * mean;
                mRunningVariance[i] = oneMinusAlpha * mRunningVariance[i] + alpha * (sumOfSquares / nMinusOne);
            }

            mUsedBatchStatistics = true;
            mLastBatchSize = batchSize;
        }

        /**
         * Single-sample backward pass for the preceding forward(). Accumulates
         * the gamma/beta gradients and writes the input deltas.
         *
         * @param outputDeltas Array of Size values, -dL/dy
         * @param inputDeltas  Array of Size values, -dL/dx
         */
        void backward(ValueType const* const outputDeltas, ValueType* inputDeltas)
        {
            backwardBatch(outputDeltas, inputDeltas, 1);
        }

        /**
         * Backward pass for the preceding forward()/forwardBatch() with the
         * same batchSize (any other batchSize is ignored). With batch
         * statistics, the exact gradient through
         * the batch mean and variance:
         *
         *   g = delta * gamma
         *   dx[n] = invStd / N * (N * g[n] - sum(g) - xhat[n] * sum(g * xhat))
         *
         * With running statistics (constants): dx = delta * gamma * invStd.
         */
        void backwardBatch(ValueType const* const outputDeltas, ValueType* inputDeltas, const size_t batchSize)
        {
            // The stored normalized values belong to the last forward pass.
            if (batchSize != mLastBatchSize)
            {
                return;
            }

            for (size_t i = 0; i < Size; ++i)
            {
                ValueType sumDelta = zero();
                ValueType sumDeltaXhat = zero();
                for (size_t sample = 0; sample < batchSize; ++sample)
                {
                    const size_t index = sample * Size + i;
                    sumDelta += outputDeltas[index];
                    sumDeltaXhat += outputDeltas[index] * mNormalized[index];
                }

                mGammaGradient[i] += sumDeltaXhat;
                mBetaGradient[i] += sumDelta;

                if (mUsedBatchStatistics)
                {
                    const ValueType n = fromInteger(static_cast<int>(batchSize));
                    const ValueType scale = mGamma[i] * mInvStd[i] / n;
                    for (size_t sample = 0; sample < batchSize; ++sample)
                    {
                        const size_t index = sample * Size + i;
                        inputDeltas[index] = scale * (n * outputDeltas[index] - sumDelta - mNormalized[index] * sumDeltaXhat);
                    }
                }
                else
                {
                    const ValueType scale = mGamma[i] * mInvStd[i];
                    for (size_t sample = 0; sample < batchSize; ++sample)
                    {
                        const size_t index = sample * Size + i;
                        inputDeltas[index] = outputDeltas[index] * scale;
                    }
                }
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
        void setRunningMean(const size_t index, const ValueType& value) { mRunningMean[index] = value; }

        ValueType getRunningVariance(const size_t index) const { return mRunningVariance[index]; }
        void setRunningVariance(const size_t index, const ValueType& value) { mRunningVariance[index] = value; }

        ValueType getGammaGradient(const size_t index) const { return mGammaGradient[index]; }
        ValueType getBetaGradient(const size_t index) const { return mBetaGradient[index]; }

        // Normalized value of feature `index` of the first (or only) sample
        // of the last forward pass.
        ValueType getNormalized(const size_t index) const { return mNormalized[index]; }
        ValueType getNormalized(const size_t sample, const size_t index) const { return mNormalized[sample * Size + index]; }

        static const size_t NumberOfFeatures = Size;
        static const size_t MaximumBatchSize = MaxBatchSize;

    private:
        void normalizeWithRunningStatistics(ValueType const* const input, ValueType* output, const size_t batchSize)
        {
            for (size_t i = 0; i < Size; ++i)
            {
                const ValueType invStd = one() / SquareRootApproximation<ValueType>::sqrt(mRunningVariance[i] + epsilon());
                mInvStd[i] = invStd;
                for (size_t sample = 0; sample < batchSize; ++sample)
                {
                    const size_t index = sample * Size + i;
                    mNormalized[index] = (input[index] - mRunningMean[i]) * invStd;
                    output[index] = mGamma[i] * mNormalized[index] + mBeta[i];
                }
            }
        }

        ValueType mGamma[Size];
        ValueType mBeta[Size];
        ValueType mRunningMean[Size];
        ValueType mRunningVariance[Size];
        ValueType mGammaGradient[Size];
        ValueType mBetaGradient[Size];
        ValueType mInvStd[Size];
        ValueType mNormalized[MaxBatchSize * Size];
        bool mTraining;
        bool mUsedBatchStatistics;
        size_t mLastBatchSize;

        // Construct a ValueType representing the integer v. FPU-free for QValue:
        // QValue(int) treats its argument as the raw fixed-point bit pattern,
        // so we must use the (fixed, fractional) constructor instead.
        template<typename T = ValueType>
        static typename tinymind::enable_if<tinymind::is_floating_point<T>::value, T>::type
        fromInteger(const int v)
        {
            return static_cast<T>(v);
        }

        template<typename T = ValueType>
        static typename tinymind::enable_if<!tinymind::is_floating_point<T>::value, T>::type
        fromInteger(const int v)
        {
            return T(static_cast<typename T::FixedPartFieldType>(v), 0u);
        }

        static ValueType zero()
        {
            static const ValueType z = fromInteger(0);
            return z;
        }

        static ValueType one()
        {
            static const ValueType o = fromInteger(1);
            return o;
        }

        static ValueType momentum()
        {
            static const ValueType m = fromInteger(static_cast<int>(MomentumPercent))
                                     / fromInteger(100);
            return m;
        }

        static ValueType epsilon()
        {
            static const ValueType e = fromInteger(static_cast<int>(EpsilonPercent))
                                     / fromInteger(100);
            return e;
        }

        static_assert(Size > 0, "BatchNorm size must be > 0.");
        static_assert(MaxBatchSize > 0, "BatchNorm maximum batch size must be > 0.");
        static_assert(MomentumPercent > 0 && MomentumPercent <= 100, "Momentum percent must be in (0, 100].");
        static_assert(EpsilonPercent > 0, "Epsilon percent must be > 0.");
    };
}
