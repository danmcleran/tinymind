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
#include <cstdint>

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
#include "include/nnproperties.hpp"   // ValueConverter

#include <cstddef>

namespace tinymind {
namespace pinn {

    // Lift a double constant into any scalar type S used by the dual machinery.
    template<typename S> struct Constant
    {
        static S of(double d) { return ValueConverter<double, S>::convertToDestinationType(d); }
    };
    template<typename V> struct Constant<Dual<V> >
    {
        static Dual<V> of(double d) { return Dual<V>(Constant<V>::of(d)); }
    };
    template<typename V, std::size_t N> struct Constant<MultiDual<V, N> >
    {
        static MultiDual<V, N> of(double d) { return MultiDual<V, N>(Constant<V>::of(d)); }
    };

    // Activation bridges: apply an activation in scalar type S. tanhValue /
    // sigmoidValue dispatch correctly for double (std), QValue (LUT), and dual
    // types (derivative-aware), so forwardAs differentiates as well as infers.
    struct TanhActivation
    {
        template<typename S> static S apply(const S& z) { return DualScalarActivation<S>::tanhValue(z); }
    };
    struct SigmoidActivation
    {
        template<typename S> static S apply(const S& z) { return DualScalarActivation<S>::sigmoidValue(z); }
    };
    struct LinearActivation
    {
        template<typename S> static S apply(const S& z) { return z; }
    };

    // Network-value -> double, for lifting weights into S via Constant<S>.
    template<typename NV>
    inline double toDouble(const NV& v) { return ValueConverter<NV, double>::convertToDestinationType(v); }
    inline double toDouble(double v) { return v; }

    /**
     * Re-evaluate a stock single-hidden-layer feed-forward NeuralNetwork in an
     * arbitrary scalar type S, reading its trained weights through the public
     * getters. This is ADDITIVE -- it does not touch the network's own
     * feedForward / trainNetwork / storage, so existing uses and the byte-exact
     * golden/embedded regressions are unaffected. With S = double it reproduces
     * getLearnedValues; with S = Dual<double> (or nested) it yields the network's
     * input derivatives (du/dx, d2u/dx2) for a PDE residual.
     *
     * HiddenAct / OutputAct must match the network's transfer-function policy
     * (e.g. TanhActivation hidden + LinearActivation output for a PINN field).
     * The net is taken by non-const reference because the hidden-layer getters
     * are non-const in the current library.
     */
    template<typename S, typename HiddenAct, typename OutputAct, typename NetT>
    void forwardAs(NetT& net, const S* inputs, S* outputs)
    {
        // Supports any number of hidden layers of UNIFORM width (e.g.
        // MultilayerPerceptron). Variable-width HiddenLayers<a,b,...> nets do
        // not expose per-layer widths publicly and are not supported -- validate
        // forwardAs<double> against getLearnedValues for your topology.
        const std::size_t NI = NetT::NumberOfInputLayerNeurons;
        const std::size_t NH = NetT::NumberOfHiddenLayerNeurons; // uniform width
        const std::size_t NL = NetT::NeuralNetworkNumberOfHiddenLayers;
        const std::size_t NO = NetT::NumberOfOutputLayerNeurons;

        S bufA[NetT::NumberOfHiddenLayerNeurons];
        S bufB[NetT::NumberOfHiddenLayerNeurons];
        S* cur  = bufA;
        S* prev = bufB;

        // First hidden layer from the inputs.
        for (std::size_t j = 0; j < NH; ++j)
        {
            S z = Constant<S>::of(toDouble(net.getInputLayerBiasNeuronWeightForConnection(j)));
            for (std::size_t i = 0; i < NI; ++i)
                z = z + Constant<S>::of(toDouble(net.getInputLayerWeightForNeuronAndConnection(i, j))) * inputs[i];
            cur[j] = HiddenAct::template apply<S>(z);
        }

        // Subsequent hidden layers: weights from layer L-1 carry index L-1.
        for (std::size_t L = 1; L < NL; ++L)
        {
            S* tmp = prev; prev = cur; cur = tmp;
            for (std::size_t m = 0; m < NH; ++m)
            {
                S z = Constant<S>::of(toDouble(net.getHiddenLayerBiasNeuronWeightForConnection(L - 1, m)));
                for (std::size_t n = 0; n < NH; ++n)
                    z = z + Constant<S>::of(toDouble(net.getHiddenLayerWeightForNeuronAndConnection(L - 1, n, m))) * prev[n];
                cur[m] = HiddenAct::template apply<S>(z);
            }
        }

        // Output from the last hidden layer (index NL-1).
        for (std::size_t k = 0; k < NO; ++k)
        {
            S z = Constant<S>::of(toDouble(net.getHiddenLayerBiasNeuronWeightForConnection(NL - 1, k)));
            for (std::size_t j = 0; j < NH; ++j)
                z = z + Constant<S>::of(toDouble(net.getHiddenLayerWeightForNeuronAndConnection(NL - 1, j, k))) * cur[j];
            outputs[k] = OutputAct::template apply<S>(z);
        }
        (void)prev;
    }

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
