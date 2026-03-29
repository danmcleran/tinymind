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

namespace tinymind {
    /**
     * Early stopping monitor for training loops.
     *
     * Tracks training error and signals when improvement has stalled,
     * allowing the training loop to exit early and save compute cycles.
     * This is especially valuable on power-constrained embedded targets.
     *
     * Usage:
     *   EarlyStopping<double> stopper;
     *   for (int i = 0; i < maxIterations; ++i) {
     *       nn.feedForward(inputs);
     *       double error = nn.calculateError(targets);
     *       if (stopper.shouldStop(error)) break;
     *       nn.trainNetwork(targets);
     *   }
     *
     * @tparam ValueType The error value type (must support comparison operators)
     * @tparam Patience  Number of consecutive non-improving steps before stopping
     */
    template<typename ValueType, size_t Patience = 100>
    struct EarlyStopping
    {
        EarlyStopping() : mBestError(), mPatienceCounter(0), mFirstStep(true)
        {
        }

        /**
         * Report the current error and check if training should stop.
         * @param error Current training or validation error
         * @return true if training should stop (no improvement for Patience steps)
         */
        bool shouldStop(const ValueType& error)
        {
            if (mFirstStep)
            {
                mBestError = error;
                mFirstStep = false;
                mPatienceCounter = 0;
                return false;
            }

            if (error < mBestError)
            {
                mBestError = error;
                mPatienceCounter = 0;
                return false;
            }

            ++mPatienceCounter;
            return (mPatienceCounter >= Patience);
        }

        /**
         * Reset the monitor to its initial state.
         */
        void reset()
        {
            mBestError = ValueType();
            mPatienceCounter = 0;
            mFirstStep = true;
        }

        /**
         * Get the best error observed so far.
         */
        ValueType getBestError() const
        {
            return mBestError;
        }

        /**
         * Get the current patience counter value.
         */
        size_t getPatienceCounter() const
        {
            return mPatienceCounter;
        }

    private:
        ValueType mBestError;
        size_t mPatienceCounter;
        bool mFirstStep;
    };
}
