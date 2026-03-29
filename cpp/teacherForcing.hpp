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
#include <cstdlib>

namespace tinymind {
    /**
     * Scheduled sampling for recurrent network training.
     *
     * During auto-regressive training, the model's own predictions are
     * sometimes substituted for ground truth inputs. This teaches the
     * model to recover from its own errors, directly addressing the
     * phase-drift problem documented in LSTM.md.
     *
     * The teacher forcing ratio starts at 1.0 (always use ground truth)
     * and linearly decays toward 0.0 (always use model prediction) over
     * a configurable number of steps.
     *
     * Usage:
     *   ScheduledSampling<double> sampler(10000); // decay over 10000 steps
     *   for each epoch:
     *       nn.resetState();
     *       for t = 0 to sequenceLength:
     *           nn.feedForward(input);
     *           nn.getLearnedValues(prediction);
     *           nn.trainNetwork(target);
     *           // Choose next input: ground truth or model prediction
     *           nextInput = sampler.selectInput(groundTruth[t+1], prediction[0]);
     *           sampler.step();
     *
     * @tparam ValueType The network value type
     */
    template<typename ValueType>
    struct ScheduledSampling
    {
        /**
         * @param totalDecaySteps Number of training steps over which the
         *        teacher forcing ratio decays from 1.0 to 0.0.
         *        After this many steps, the model always uses its own predictions.
         */
        ScheduledSampling(const size_t totalDecaySteps)
            : mTotalDecaySteps(totalDecaySteps), mCurrentStep(0)
        {
        }

        /**
         * Select between ground truth and model prediction.
         *
         * With probability equal to the current teacher forcing ratio,
         * returns the ground truth value. Otherwise returns the model's
         * prediction.
         *
         * @param groundTruth The true next value from the training data
         * @param prediction The model's predicted next value
         * @return Either groundTruth or prediction
         */
        ValueType selectInput(const ValueType& groundTruth, const ValueType& prediction) const
        {
            // Teacher forcing ratio: 1.0 at start, decays to 0.0
            if (mCurrentStep >= mTotalDecaySteps)
            {
                return prediction;
            }

            // Use integer random for embedded compatibility (no <random> needed)
            const unsigned threshold = static_cast<unsigned>(
                ((mTotalDecaySteps - mCurrentStep) * 1000U) / mTotalDecaySteps);
            const unsigned roll = static_cast<unsigned>(rand() % 1000U);

            if (roll < threshold)
            {
                return groundTruth;
            }
            return prediction;
        }

        /**
         * Advance one training step, reducing the teacher forcing ratio.
         */
        void step()
        {
            if (mCurrentStep < mTotalDecaySteps)
            {
                ++mCurrentStep;
            }
        }

        /**
         * Reset to the beginning (full teacher forcing).
         */
        void reset()
        {
            mCurrentStep = 0;
        }

        /**
         * Get the current teacher forcing ratio (1.0 = always truth, 0.0 = always prediction).
         */
        double getTeacherForcingRatio() const
        {
            if (mCurrentStep >= mTotalDecaySteps)
            {
                return 0.0;
            }
            return static_cast<double>(mTotalDecaySteps - mCurrentStep) / static_cast<double>(mTotalDecaySteps);
        }

        /**
         * Get the current step count.
         */
        size_t getCurrentStep() const
        {
            return mCurrentStep;
        }

    private:
        size_t mTotalDecaySteps;
        size_t mCurrentStep;
    };
}
