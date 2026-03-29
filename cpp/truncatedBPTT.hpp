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
     * Truncated Backpropagation Through Time (TBPTT) training utility.
     *
     * Accumulates forward passes over a window of K timesteps before
     * performing a single weight update. This enables the recurrent
     * hidden state to carry information across K timesteps before
     * training, producing better temporal gradient flow than the default
     * single-step train-per-timestep approach.
     *
     * The window size K is a compile-time template parameter, so the
     * stored targets array has known size with zero dynamic allocation.
     *
     * Usage:
     *   TruncatedBPTT<LstmType, 5> trainer;  // window of 5 timesteps
     *   for each epoch:
     *       nn.resetState();
     *       trainer.reset();
     *       for t = 0 to sequenceLength:
     *           trainer.step(nn, &input[t], &target[t]);
     *       trainer.flush(nn);  // train on any remaining steps
     *
     * @tparam NeuralNetworkType Any recurrent TinyMind network (LSTM, GRU, Elman)
     * @tparam WindowSize Number of timesteps to accumulate before updating weights
     */
    template<typename NeuralNetworkType, size_t WindowSize>
    struct TruncatedBPTT
    {
        typedef typename NeuralNetworkType::NeuralNetworkValueType ValueType;
        static const size_t NumberOfOutputs = NeuralNetworkType::NumberOfOutputLayerNeurons;
        static const size_t NumberOfInputs = NeuralNetworkType::NumberOfInputLayerNeurons;

        TruncatedBPTT() : mStepCount(0)
        {
        }

        /**
         * Feed one timestep. When the window is full, performs a weight
         * update using the most recent target values.
         *
         * @param nn     The recurrent network (state is preserved across steps)
         * @param input  Input values for this timestep
         * @param target Target values for this timestep
         */
        void step(NeuralNetworkType& nn, ValueType const* const input, ValueType const* const target)
        {
            // Feed forward (accumulates recurrent state)
            nn.feedForward(input);

            // Store target for potential training
            for (size_t o = 0; o < NumberOfOutputs; ++o)
            {
                mTargets[mStepCount * NumberOfOutputs + o] = target[o];
            }

            ++mStepCount;

            // When window is full, train on the last target
            if (mStepCount >= WindowSize)
            {
                nn.trainNetwork(&mTargets[(WindowSize - 1) * NumberOfOutputs]);
                mStepCount = 0;
            }
        }

        /**
         * Train on any remaining accumulated steps when the sequence
         * doesn't divide evenly by the window size.
         * Call at the end of each sequence/epoch.
         *
         * @param nn The recurrent network
         */
        void flush(NeuralNetworkType& nn)
        {
            if (mStepCount > 0)
            {
                nn.trainNetwork(&mTargets[(mStepCount - 1) * NumberOfOutputs]);
                mStepCount = 0;
            }
        }

        /**
         * Reset the step counter without training. Call when resetting
         * the network state between epochs.
         */
        void reset()
        {
            mStepCount = 0;
        }

        /**
         * Get the current number of accumulated timesteps.
         */
        size_t getStepCount() const
        {
            return mStepCount;
        }

    private:
        size_t mStepCount;
        ValueType mTargets[WindowSize * NumberOfOutputs];

        static_assert(WindowSize > 0, "Window size must be > 0.");
    };
}
