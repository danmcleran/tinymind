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

// Reusable Physics-Informed Neural Network building blocks:
//   - PinnMlp: a small fully-connected tanh MLP whose forward pass is templated
//     on the scalar type, so the SAME network differentiates w.r.t. its inputs
//     (Dual / nested Dual -> du/dx, d2u/dx2), differentiates w.r.t. its weights
//     (MultiDual -> the loss gradient), and runs plain inference (double or
//     fixed-point QValue). It addresses "make the network Dual-differentiable":
//     the stock NeuralNet<> cannot be (its activation policies are QValue-typed),
//     so this is a PINN-appropriate net that is differentiable by construction.
//   - sgdStep: a PDE-agnostic momentum-SGD step that gets the EXACT loss
//     gradient w.r.t. all weights in one forward pass via MultiDual, then
//     updates. The PDE-specific residual/loss is supplied as a functor with a
//     templated operator() so it can be evaluated in any scalar type.
//
// Activation is dispatched through DualScalarActivation<S>::tanhValue, which
// resolves to std::tanh (float/double), the lookup table (QValue), or recursive
// dual tanh (Dual / MultiDual) -- so forward<S> compiles for every S the rest of
// the dual machinery supports.

#include "include/tinymind_platform.hpp"
#include "dual.hpp"
#include "dualActivations.hpp"
#include "multidual.hpp"

#include <cstddef>

namespace tinymind {
namespace pinn {

    /**
     * Fully-connected MLP: NumInputs -> NumHidden (tanh) -> NumOutputs (linear).
     * Weights are caller-owned in a flat array, layout:
     *   [W1: NumHidden*NumInputs] [b1: NumHidden]
     *   [W2: NumOutputs*NumHidden] [b2: NumOutputs]
     * forward() is templated on the scalar type; pass `double` for inference,
     * `Dual<...>` for input derivatives, `MultiDual<double,NumParams>` for the
     * weight gradient.
     */
    template<std::size_t NumInputs, std::size_t NumHidden, std::size_t NumOutputs>
    struct PinnMlp
    {
        static const std::size_t W1Count = NumHidden * NumInputs;
        static const std::size_t W2Count = NumOutputs * NumHidden;
        static const std::size_t NumParams = W1Count + NumHidden + W2Count + NumOutputs;

        template<typename S>
        static void forward(const S* p, const S* inputs, S* outputs)
        {
            const S* W1 = p;
            const S* b1 = W1 + W1Count;
            const S* W2 = b1 + NumHidden;
            const S* b2 = W2 + W2Count;

            S hidden[NumHidden];
            for (std::size_t h = 0; h < NumHidden; ++h)
            {
                S z = b1[h];
                for (std::size_t i = 0; i < NumInputs; ++i)
                    z = z + W1[h * NumInputs + i] * inputs[i];
                hidden[h] = DualScalarActivation<S>::tanhValue(z);
            }

            for (std::size_t o = 0; o < NumOutputs; ++o)
            {
                S acc = b2[o];
                for (std::size_t h = 0; h < NumHidden; ++h)
                    acc = acc + W2[o * NumHidden + h] * hidden[h];
                outputs[o] = acc;
            }
        }
    };

    /**
     * One momentum-SGD step on a parameter vector of length NumParams using the
     * exact loss gradient from a single MultiDual forward pass.
     *
     * `loss` is any object with `template<typename S> S operator()(const S* p) const`
     * returning the scalar loss evaluated in scalar type S. Returns the loss
     * value at the pre-update parameters.
     */
    template<std::size_t NumParams, typename LossFn>
    double sgdStep(double* params, double* velocity,
                   double learningRate, double momentum, const LossFn& loss)
    {
        typedef MultiDual<double, NumParams> M;

        M mp[NumParams];
        for (std::size_t i = 0; i < NumParams; ++i)
            mp[i] = M::seed(params[i], i, 1.0);

        const M L = loss(&mp[0]);            // value + full gradient, one pass

        for (std::size_t i = 0; i < NumParams; ++i)
        {
            velocity[i] = momentum * velocity[i] - learningRate * L.grad[i];
            params[i] += velocity[i];
        }
        return L.value;
    }

} // namespace pinn
} // namespace tinymind
