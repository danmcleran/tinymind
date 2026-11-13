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
#include <cstdint>

#include "include/tinymind_rounding.hpp"
#include "include/tinymind_traits.hpp"
#include "activationFunctions.hpp"

namespace tinymind {
    namespace detail
    {
        // 1.0 in ValueType: QValue(1, 0) for fixed point, a cast for float.
        template<typename ValueType, bool IsFloat = tinymind::is_floating_point<ValueType>::value>
        struct HeadUnit
        {
            static ValueType one()
            {
                return ValueType(1, 0);
            }

            static ValueType zero()
            {
                return ValueType(0, 0);
            }
        };

        template<typename ValueType>
        struct HeadUnit<ValueType, true>
        {
            static ValueType one()
            {
                return static_cast<ValueType>(1);
            }

            static ValueType zero()
            {
                return static_cast<ValueType>(0);
            }
        };

        // exp(x) for x <= 0, the softmax's shifted logits. The fixed-point
        // exp table spans [MIN_X_TABLE_VALUE, MAX_X_TABLE_VALUE] = [-6, 6] and
        // clamps below, so every class more than 6 behind the largest would
        // keep e^-6 ~ 0.0025 of the probability: the softmax never reaches
        // a one-hot, the gradient never reaches 0, and the logits of a well
        // separated head grow without bound until the format wraps. Below
        // the table the true value is under 0.0025, so it is taken as 0.
        template<typename ValueType, bool IsFloat = tinymind::is_floating_point<ValueType>::value>
        struct HeadExp
        {
            static ValueType exp(const ValueType& value)
            {
                if (value <= ValueType(MIN_X_TABLE_VALUE, 0))
                {
                    return HeadUnit<ValueType>::zero();
                }

                return ActivationExp<ValueType>::exp(value);
            }
        };

        template<typename ValueType>
        struct HeadExp<ValueType, true>
        {
            static ValueType exp(const ValueType& value)
            {
                return ActivationExp<ValueType>::exp(value);
            }
        };
    }

    /**
     * Loss policies for DenseHead. Each turns the head's logits and a target
     * into output deltas, following the library's convention delta = -dL/dz:
     * a parameter moves by +learningRate x delta x input.
     */

    /**
     * Softmax over the logits with cross-entropy loss, the usual classifier
     * head. delta = t - softmax(z). The largest logit is subtracted before
     * exp, so every exp argument is <= 0 and every exp lies in (0, 1]; a
     * fixed-point format does not overflow however large the logits grow.
     * In fixed point a class more than 6 behind the largest logit, beyond
     * the exp table, gets probability 0 (its true value is under 0.0025).
     * QValue types use the exp lookup table (define TINYMIND_USE_EXP_<I>_<F>
     * and link lookupTables.cpp); float and double need TINYMIND_ENABLE_FLOAT
     * and TINYMIND_ENABLE_STD.
     */
    template<typename ValueType>
    struct SoftmaxCrossEntropyHeadLoss
    {
        static void probabilities(ValueType const* const logits, ValueType* probabilities, const size_t count)
        {
            ValueType largest = logits[0];
            for (size_t k = 1; k < count; ++k)
            {
                if (logits[k] > largest)
                {
                    largest = logits[k];
                }
            }

            ValueType sum = detail::HeadUnit<ValueType>::zero();
            for (size_t k = 0; k < count; ++k)
            {
                probabilities[k] = detail::HeadExp<ValueType>::exp(logits[k] - largest);
                sum += probabilities[k];
            }

            // exp(0) = 1 is in the sum, so it is never 0.
            if (sum != detail::HeadUnit<ValueType>::zero())
            {
                for (size_t k = 0; k < count; ++k)
                {
                    probabilities[k] /= sum;
                }
            }
        }

        static void outputDeltas(ValueType const* const logits, ValueType const* const targets, ValueType* deltas, const size_t count)
        {
            probabilities(logits, deltas, count);
            for (size_t k = 0; k < count; ++k)
            {
                deltas[k] = targets[k] - deltas[k];
            }
        }
    };

    /**
     * Linear output with squared-error loss L = 0.5 x sum((t - z)^2), for a
     * regression head or a classifier that regresses a one-hot target.
     * delta = t - z. Arithmetic only: no lookup table, no <cmath>.
     */
    template<typename ValueType>
    struct SquaredErrorHeadLoss
    {
        static void outputDeltas(ValueType const* const logits, ValueType const* const targets, ValueType* deltas, const size_t count)
        {
            for (size_t k = 0; k < count; ++k)
            {
                deltas[k] = targets[k] - logits[k];
            }
        }
    };

    /**
     * A trainable dense layer to put on top of a frozen model: head-only
     * fine-tuning.
     *
     * The frozen model (an int8 pipeline, a QValue network, anything) turns a
     * sample into an embedding of NumberOfInputs values; this head maps the
     * embedding to NumberOfOutputs logits and learns on the device from
     * labelled samples. Nothing is backpropagated through the frozen model,
     * so training needs no stored activations and no gradients for it: the
     * training state is this layer's weights and one embedding. The update is
     * fused into the backward pass, so the head keeps no gradient buffer
     * either; its RAM is (NumberOfInputs + 1) x NumberOfOutputs values.
     *
     * Usage:
     *   DenseHead<QValue<16, 16, true>, 32, 6> head;
     *   head.loadWeights(shippedWeights, shippedBiases);  // or start from 0
     *   backbone(sample, embedding);
     *   head.trainClass(embedding, label, learningRate);   // on device
     *   const size_t predicted = head.predict(embedding);
     *
     * To undo an adaptation, reload the shipped weights with loadWeights():
     * keep them in flash rather than spend RAM on a copy.
     *
     * The SGD step is plain gradient descent: learningRate x (delta x input)
     * rounded to nearest by multiplyRoundNearest, as every trainable layer in
     * the library does (see tinymind_rounding.hpp). There is no momentum.
     * Overflow follows ValueType: QValue wraps by default, so keep the
     * embedding and the logits well inside the format's range (z-scored
     * inputs and a Q16.16 head are the working recipe).
     *
     * backward() returns dL/d(embedding) x -1 (the same delta convention), for
     * a caller that also wants to adapt something in front of the head, such
     * as the biases of the frozen model's last layer.
     *
     * @tparam ValueType       QValue or float/double
     * @tparam NumberOfInputs  Size of the embedding
     * @tparam NumberOfOutputs Number of logits (classes)
     * @tparam LossPolicy      SoftmaxCrossEntropyHeadLoss (default) or SquaredErrorHeadLoss
     */
    template<
        typename ValueType,
        size_t NumberOfInputs,
        size_t NumberOfOutputs,
        typename LossPolicy = SoftmaxCrossEntropyHeadLoss<ValueType> >
    class DenseHead
    {
    public:
        static const size_t NumberOfWeights = NumberOfInputs * NumberOfOutputs;
        static const size_t NumberOfBiases = NumberOfOutputs;
        static const size_t NumberOfParameters = NumberOfWeights + NumberOfBiases;

        DenseHead()
        {
            clear();
        }

        /**
         * Set every weight and bias to 0. A zero head predicts class 0 for
         * every input until it is trained.
         */
        void clear()
        {
            for (size_t i = 0; i < NumberOfWeights; ++i)
            {
                mWeights[i] = detail::HeadUnit<ValueType>::zero();
            }
            for (size_t o = 0; o < NumberOfOutputs; ++o)
            {
                mBiases[o] = detail::HeadUnit<ValueType>::zero();
            }
        }

        template<typename RandomNumberGeneratorPolicy>
        void initializeWeights()
        {
            for (size_t i = 0; i < NumberOfWeights; ++i)
            {
                mWeights[i] = RandomNumberGeneratorPolicy::generateRandomWeight();
            }
            for (size_t o = 0; o < NumberOfOutputs; ++o)
            {
                mBiases[o] = detail::HeadUnit<ValueType>::zero();
            }
        }

        /**
         * Load every parameter, for example the shipped head from flash.
         * @param weights NumberOfOutputs x NumberOfInputs values, output-major:
         *                weights[o * NumberOfInputs + i] connects input i to output o
         * @param biases  NumberOfOutputs values
         */
        void loadWeights(ValueType const* const weights, ValueType const* const biases)
        {
            for (size_t i = 0; i < NumberOfWeights; ++i)
            {
                mWeights[i] = weights[i];
            }
            for (size_t o = 0; o < NumberOfOutputs; ++o)
            {
                mBiases[o] = biases[o];
            }
        }

        /**
         * logits[o] = bias[o] + sum_i weight[o][i] x input[i]
         */
        void forward(ValueType const* const input, ValueType* logits) const
        {
            for (size_t o = 0; o < NumberOfOutputs; ++o)
            {
                ValueType const* const row = &mWeights[o * NumberOfInputs];
                ValueType sum = mBiases[o];
                for (size_t i = 0; i < NumberOfInputs; ++i)
                {
                    sum += row[i] * input[i];
                }
                logits[o] = sum;
            }
        }

        /**
         * Index of the largest logit (the first one on a tie).
         */
        size_t predict(ValueType const* const input) const
        {
            ValueType logits[NumberOfOutputs];
            forward(input, &logits[0]);

            return argmax(&logits[0]);
        }

        /**
         * Output deltas (t - y under the loss policy) for logits from forward().
         */
        void outputDeltas(ValueType const* const logits, ValueType const* const targets, ValueType* deltas) const
        {
            LossPolicy::outputDeltas(logits, targets, deltas, NumberOfOutputs);
        }

        /**
         * inputDeltas[i] = sum_o weight[o][i] x outputDeltas[o], with the
         * weights as they are now. Call it before update() when a caller
         * adapts what lies in front of the head.
         */
        void backward(ValueType const* const outputDeltas, ValueType* inputDeltas) const
        {
            for (size_t i = 0; i < NumberOfInputs; ++i)
            {
                inputDeltas[i] = detail::HeadUnit<ValueType>::zero();
            }
            for (size_t o = 0; o < NumberOfOutputs; ++o)
            {
                ValueType const* const row = &mWeights[o * NumberOfInputs];
                const ValueType delta = outputDeltas[o];
                for (size_t i = 0; i < NumberOfInputs; ++i)
                {
                    inputDeltas[i] += row[i] * delta;
                }
            }
        }

        /**
         * One SGD step: weight[o][i] += learningRate x (delta[o] x input[i]),
         * bias[o] += learningRate x delta[o], each product with the learning
         * rate rounded to nearest.
         */
        void update(ValueType const* const outputDeltas, ValueType const* const input, const ValueType& learningRate)
        {
            for (size_t o = 0; o < NumberOfOutputs; ++o)
            {
                ValueType* const row = &mWeights[o * NumberOfInputs];
                const ValueType delta = outputDeltas[o];
                for (size_t i = 0; i < NumberOfInputs; ++i)
                {
                    row[i] += multiplyRoundNearest(learningRate, static_cast<ValueType>(delta * input[i]));
                }
                mBiases[o] += multiplyRoundNearest(learningRate, delta);
            }
        }

        /**
         * Forward, loss, optional backward and update on one sample.
         * @param inputDeltas If not null, receives backward() of this sample's
         *                    deltas, computed before the update.
         * @return The prediction made before the update, for running accuracy.
         */
        size_t train(ValueType const* const input, ValueType const* const targets, const ValueType& learningRate, ValueType* inputDeltas = nullptr)
        {
            ValueType logits[NumberOfOutputs];
            ValueType deltas[NumberOfOutputs];

            forward(input, &logits[0]);
            const size_t predicted = argmax(&logits[0]);
            LossPolicy::outputDeltas(&logits[0], targets, &deltas[0], NumberOfOutputs);
            if (inputDeltas != nullptr)
            {
                backward(&deltas[0], inputDeltas);
            }
            update(&deltas[0], input, learningRate);

            return predicted;
        }

        /**
         * train() with a one-hot target for classIndex.
         */
        size_t trainClass(ValueType const* const input, const size_t classIndex, const ValueType& learningRate, ValueType* inputDeltas = nullptr)
        {
            ValueType targets[NumberOfOutputs];
            for (size_t o = 0; o < NumberOfOutputs; ++o)
            {
                targets[o] = (o == classIndex) ? detail::HeadUnit<ValueType>::one() : detail::HeadUnit<ValueType>::zero();
            }

            return train(input, &targets[0], learningRate, inputDeltas);
        }

        ValueType getWeight(const size_t output, const size_t input) const
        {
            return mWeights[output * NumberOfInputs + input];
        }

        void setWeight(const size_t output, const size_t input, const ValueType& value)
        {
            mWeights[output * NumberOfInputs + input] = value;
        }

        ValueType getBias(const size_t output) const
        {
            return mBiases[output];
        }

        void setBias(const size_t output, const ValueType& value)
        {
            mBiases[output] = value;
        }

    private:
        static size_t argmax(ValueType const* const logits)
        {
            size_t best = 0;
            for (size_t o = 1; o < NumberOfOutputs; ++o)
            {
                if (logits[o] > logits[best])
                {
                    best = o;
                }
            }

            return best;
        }

        ValueType mWeights[NumberOfWeights];
        ValueType mBiases[NumberOfBiases];

        static_assert(NumberOfInputs > 0, "A head needs at least one input.");
        static_assert(NumberOfOutputs > 0, "A head needs at least one output.");
    };
}
