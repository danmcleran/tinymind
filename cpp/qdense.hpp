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
#include "qaffine.hpp"

#include <cstddef>
#include <cstdint>

/*
 * Quantized fully-connected (Dense) layer.
 *
 * Standalone layer that mirrors the affine quantization model used by
 * TFLite reference / CMSIS-NN kernels. It does NOT integrate with the
 * NeuralNet<> template; weights, biases, and the per-output Requantizer
 * are caller-owned and the forward pass is a single function call.
 *
 * Type contract (matches the canonical int8 deployment shape):
 *
 *   InputStorage  = int8_t   asymmetric activations (zero_point baked into
 *                            input_zero_point at calibration)
 *   WeightStorage = int8_t   symmetric weights (zero_point = 0)
 *   AccumType     = int32_t  bias and accumulator
 *   OutputStorage = int8_t   asymmetric output (handled inside Requantizer)
 *
 * The forward pass evaluates:
 *
 *   acc[o] = bias[o] + sum_i weight[o,i] * (input[i] - input_zero_point)
 *   output[o] = Requantizer.apply(acc[o])
 *
 * Bias is held in the accumulator type with effective scale equal to
 * input_scale * weight_scale, so it is added directly to the partial sum
 * without further rescaling. The Requantizer applies the
 *   effective_ratio = (input_scale * weight_scale) / output_scale
 * decomposition computed at calibration time.
 *
 * weights is a row-major buffer of shape [NumOutputs * NumInputs].
 * biases may be nullptr to indicate "no bias"; in that case the
 * accumulator starts at zero.
 *
 * Pure integer at runtime; no float, no <cmath>, no stdlib. Safe to
 * compile in the freestanding (FLOAT=0, STD=0) configuration.
 */

namespace tinymind {

    template<typename InputStorage_, typename WeightStorage_, typename AccumType_,
             typename OutputStorage_, std::size_t NumInputs_, std::size_t NumOutputs_>
    struct QDense
    {
        typedef InputStorage_ InputType;
        typedef WeightStorage_ WeightType;
        typedef AccumType_ AccumulatorType;
        typedef OutputStorage_ OutputType;

        static constexpr std::size_t InputLength = NumInputs_;
        static constexpr std::size_t OutputLength = NumOutputs_;

        const WeightType* weights;
        const AccumulatorType* biases;
        InputType input_zero_point;
        Requantizer<AccumulatorType, OutputType> requantizer;

        void forward(const InputType* input, OutputType* output) const
        {
            for (std::size_t o = 0; o < NumOutputs_; ++o)
            {
                AccumulatorType acc = (biases != nullptr)
                    ? biases[o]
                    : static_cast<AccumulatorType>(0);

                const WeightType* w_row = weights + o * NumInputs_;
                for (std::size_t i = 0; i < NumInputs_; ++i)
                {
                    const AccumulatorType x =
                        static_cast<AccumulatorType>(input[i]) -
                        static_cast<AccumulatorType>(input_zero_point);
                    acc += static_cast<AccumulatorType>(w_row[i]) * x;
                }

                output[o] = requantizer.apply(acc);
            }
        }
    };

} // namespace tinymind
