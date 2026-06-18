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

namespace tinymind {
    /**
     * Single affine layer: OutputDim = W * InputDim + bias.
     *
     * Default expert (and internal router) for MixtureOfExperts. It is the
     * minimal layer that satisfies the Expert contract below, so it doubles as
     * a ready-to-use expert and as the linear gating network. Weights are
     * stored row-major, output-major: weight(o, i) at index o * InputDim + i.
     *
     * @tparam ValueType Numeric type (QValue or float/double)
     * @tparam InputDim  Number of inputs
     * @tparam OutputDim Number of outputs
     */
    template<
        typename ValueType,
        size_t InputDim,
        size_t OutputDim>
    class LinearExpert
    {
    public:
        static const size_t InputSize = InputDim;
        static const size_t OutputSize = OutputDim;
        static const size_t TotalWeights = InputDim * OutputDim;
        static const size_t TotalBiases = OutputDim;

        LinearExpert()
        {
            for (size_t i = 0; i < TotalWeights; ++i)
            {
                mWeights[i] = ValueType(0);
            }
            for (size_t o = 0; o < TotalBiases; ++o)
            {
                mBiases[o] = ValueType(0);
            }
        }

        /**
         * Initialize weights with values from a random number generator.
         */
        template<typename RandomNumberGeneratorPolicy>
        void initializeWeights()
        {
            for (size_t i = 0; i < TotalWeights; ++i)
            {
                mWeights[i] = RandomNumberGeneratorPolicy::generateRandomWeight();
            }
            for (size_t o = 0; o < TotalBiases; ++o)
            {
                mBiases[o] = ValueType(0);
            }
        }

        /**
         * Forward pass: output[o] = bias[o] + sum_i weight(o, i) * input[i].
         * @param input  Array of InputDim values
         * @param output Array of OutputDim values
         */
        void forward(ValueType const* const input, ValueType* output) const
        {
            for (size_t o = 0; o < OutputDim; ++o)
            {
                ValueType sum = mBiases[o];
                const size_t row = o * InputDim;
                for (size_t i = 0; i < InputDim; ++i)
                {
                    sum += mWeights[row + i] * input[i];
                }
                output[o] = sum;
            }
        }

        // Flat weight accessors for serialization
        ValueType getWeight(const size_t index) const { return mWeights[index]; }
        void setWeight(const size_t index, const ValueType& value) { mWeights[index] = value; }

        ValueType getBias(const size_t index) const { return mBiases[index]; }
        void setBias(const size_t index, const ValueType& value) { mBiases[index] = value; }

        /**
         * Structured weight accessor.
         * @param outputIndex Output index [0, OutputDim)
         * @param inputIndex  Input index  [0, InputDim)
         */
        ValueType getWeightAt(const size_t outputIndex, const size_t inputIndex) const
        {
            return mWeights[outputIndex * InputDim + inputIndex];
        }

        void setWeightAt(const size_t outputIndex, const size_t inputIndex, const ValueType& value)
        {
            mWeights[outputIndex * InputDim + inputIndex] = value;
        }

    private:
        ValueType mWeights[TotalWeights];
        ValueType mBiases[TotalBiases];

        static_assert(InputDim > 0, "Input dimension must be > 0.");
        static_assert(OutputDim > 0, "Output dimension must be > 0.");
    };

    /**
     * Mixture-of-Experts layer with top-1 (Switch-style) routing.
     *
     * A small linear router scores the input against NumExperts experts; the
     * single highest-scoring expert runs and produces the output. Only one
     * expert executes per call, so active compute equals one expert even though
     * all NumExperts experts are resident in memory. This trades storage
     * (sum of all expert weights) for flat per-inference compute -- the right
     * trade when flash is plentiful but cycles/energy are tight and the task is
     * heterogeneous enough for the router to separate input regimes.
     *
     * Routing is an argmax over router logits. argmax over raw logits is
     * identical to argmax over their softmax (softmax is monotonic), so no
     * softmax is needed to pick the expert, and the choice is robust to
     * quantization noise -- it only flips on near-ties, where the two experts
     * agree anyway. forward() therefore performs pure dispatch with no gate
     * scaling; fold any gate weight into the expert during training, or apply
     * it externally using the logits from route().
     *
     * This phase-1 layer is inference-and-weight-loading oriented. Joint
     * training (which must contend with the non-differentiable argmax and
     * load balancing) is a host/PyTorch concern -- train there, then load the
     * frozen router and expert weights here.
     *
     * Experts are pluggable: ExpertType is a template taking
     * <ValueType, InputDim, OutputDim>. Any layer adapted to that shape works
     * (the default is LinearExpert). The expert contract is:
     *   - constructs default
     *   - exposes static InputSize == InputDim and OutputSize == OutputDim
     *   - void forward(ValueType const* input, ValueType* output)
     *
     * Usage:
     *   MixtureOfExperts<double, 8, 4, 3> moe;       // 8 in, 4 out, 3 LinearExpert experts
     *   moe.forward(input, output);                  // argmax router -> 1 expert
     *   size_t which = moe.getLastSelectedExpert();
     *
     * @tparam ValueType  Numeric type (QValue or float/double)
     * @tparam InputDim   Input feature dimension
     * @tparam OutputDim  Output feature dimension
     * @tparam NumExperts Number of experts (all resident; one runs per call)
     * @tparam ExpertType Expert layer template <ValueType, InputDim, OutputDim>
     */
    template<
        typename ValueType,
        size_t InputDim,
        size_t OutputDim,
        size_t NumExperts,
        template<typename, size_t, size_t> class ExpertType = LinearExpert>
    class MixtureOfExperts
    {
    public:
        typedef ExpertType<ValueType, InputDim, OutputDim> Expert;
        typedef LinearExpert<ValueType, InputDim, NumExperts> Router;

        static const size_t InputSize = InputDim;
        static const size_t OutputSize = OutputDim;
        static const size_t NumberOfExperts = NumExperts;

        MixtureOfExperts() : mLastExpert(0)
        {
        }

        /**
         * Initialize router and all expert weights from a random number
         * generator policy.
         */
        template<typename RandomNumberGeneratorPolicy>
        void initializeWeights()
        {
            mRouter.template initializeWeights<RandomNumberGeneratorPolicy>();
            for (size_t e = 0; e < NumExperts; ++e)
            {
                mExperts[e].template initializeWeights<RandomNumberGeneratorPolicy>();
            }
        }

        /**
         * Forward pass: route to the top-1 expert and run it.
         * @param input  Array of InputDim values
         * @param output Array of OutputDim values, filled by the selected expert
         */
        void forward(ValueType const* const input, ValueType* output)
        {
            ValueType logits[NumExperts];
            mRouter.forward(input, logits);
            mLastExpert = argMax(logits);
            mExperts[mLastExpert].forward(input, output);
        }

        /**
         * Compute router logits and the selected expert without running it.
         * Useful for inspecting routing or applying an external gate weight.
         * @param input  Array of InputDim values
         * @param logits Array of NumExperts values, filled with router scores
         * @return index of the argmax (selected) expert
         */
        size_t route(ValueType const* const input, ValueType* logits) const
        {
            mRouter.forward(input, logits);
            return argMax(logits);
        }

        /**
         * Index of the expert selected by the most recent forward() call.
         */
        size_t getLastSelectedExpert() const { return mLastExpert; }

        // Sub-layer accessors for weight loading / serialization
        Router& router() { return mRouter; }
        const Router& router() const { return mRouter; }

        Expert& expert(const size_t index) { return mExperts[index]; }
        const Expert& expert(const size_t index) const { return mExperts[index]; }

    private:
        /**
         * Index of the maximum value over NumExperts entries. Ties resolve to
         * the lowest index (first maximum wins).
         */
        static size_t argMax(ValueType const* const values)
        {
            size_t best = 0;
            ValueType bestVal = values[0];
            for (size_t e = 1; e < NumExperts; ++e)
            {
                if (values[e] > bestVal)
                {
                    bestVal = values[e];
                    best = e;
                }
            }
            return best;
        }

        Router mRouter;
        Expert mExperts[NumExperts];
        size_t mLastExpert;

        static_assert(NumExperts > 0, "Number of experts must be > 0.");
        static_assert(InputDim > 0, "Input dimension must be > 0.");
        static_assert(OutputDim > 0, "Output dimension must be > 0.");
        static_assert(Expert::OutputSize == OutputDim,
                      "Expert OutputSize must equal MoE OutputDim.");
        static_assert(Expert::InputSize == InputDim,
                      "Expert InputSize must equal MoE InputDim.");
    };

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
    /**
     * Top-k / dense Mixture-of-Experts with softmax-gated blending.
     *
     * Unlike the top-1 MixtureOfExperts (which runs a single argmax expert),
     * this layer runs the K highest-scoring experts and returns their
     * gate-weighted sum:
     *
     *   topk     = indices of the K largest router logits
     *   g_i      = softmax(logits)_i over i in topk   (renormalized on topk)
     *   output   = sum_{i in topk} g_i * expert_i(x)
     *
     * K == 1            collapses to top-1 with gate weight 1.0 (the argmax
     *                   expert, identical to MixtureOfExperts).
     * K == NumExperts   is a dense MoE: every expert runs, softmax over all
     *                   logits weights the blend.
     *
     * The softmax requires real exponentials, so this layer is gated on
     * TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD and is meant for float /
     * double value types (the gate is computed in double). The deployable
     * pure-integer counterpart is QTopKMixtureOfExperts in qmoe.hpp, which
     * blends with fixed-point gate weights from an exp lookup table. Ties in
     * the top-k selection resolve to the lowest index.
     *
     * @tparam ValueType  float or double
     * @tparam InputDim   Input feature dimension
     * @tparam OutputDim  Output feature dimension
     * @tparam NumExperts Number of experts (all resident)
     * @tparam K          Experts blended per call (1 <= K <= NumExperts)
     * @tparam ExpertType Expert layer template <ValueType, InputDim, OutputDim>
     */
    template<
        typename ValueType,
        size_t InputDim,
        size_t OutputDim,
        size_t NumExperts,
        size_t K,
        template<typename, size_t, size_t> class ExpertType = LinearExpert>
    class TopKMixtureOfExperts
    {
    public:
        typedef ExpertType<ValueType, InputDim, OutputDim> Expert;
        typedef LinearExpert<ValueType, InputDim, NumExperts> Router;

        static const size_t InputSize = InputDim;
        static const size_t OutputSize = OutputDim;
        static const size_t NumberOfExperts = NumExperts;
        static const size_t TopK = K;

        TopKMixtureOfExperts()
        {
            for (size_t j = 0; j < K; ++j)
            {
                mLastIndices[j] = 0;
                mLastGates[j] = ValueType(0);
            }
        }

        template<typename RandomNumberGeneratorPolicy>
        void initializeWeights()
        {
            mRouter.template initializeWeights<RandomNumberGeneratorPolicy>();
            for (size_t e = 0; e < NumExperts; ++e)
            {
                mExperts[e].template initializeWeights<RandomNumberGeneratorPolicy>();
            }
        }

        /**
         * Forward pass: blend the K top-scoring experts by their softmax gate.
         * @param input  Array of InputDim values
         * @param output Array of OutputDim values (gate-weighted expert sum)
         */
        void forward(ValueType const* const input, ValueType* output)
        {
            ValueType logits[NumExperts];
            mRouter.forward(input, logits);

            selectTopK(logits, mLastIndices);

            // Softmax over the K selected logits (shifted by their max).
            double maxLogit = static_cast<double>(logits[mLastIndices[0]]);
            for (size_t j = 1; j < K; ++j)
            {
                const double v = static_cast<double>(logits[mLastIndices[j]]);
                if (v > maxLogit)
                {
                    maxLogit = v;
                }
            }
            double ex[K];
            double sum = 0.0;
            for (size_t j = 0; j < K; ++j)
            {
                ex[j] = exp_double(static_cast<double>(logits[mLastIndices[j]]) - maxLogit);
                sum += ex[j];
            }

            for (size_t o = 0; o < OutputDim; ++o)
            {
                output[o] = ValueType(0);
            }

            ValueType expertOut[OutputDim];
            for (size_t j = 0; j < K; ++j)
            {
                const double gate = (sum > 0.0) ? (ex[j] / sum) : (1.0 / K);
                mLastGates[j] = static_cast<ValueType>(gate);
                mExperts[mLastIndices[j]].forward(input, expertOut);
                for (size_t o = 0; o < OutputDim; ++o)
                {
                    output[o] += mLastGates[j] * expertOut[o];
                }
            }
        }

        size_t getSelectedExpert(const size_t rank) const { return mLastIndices[rank]; }
        ValueType getGate(const size_t rank) const { return mLastGates[rank]; }

        Router& router() { return mRouter; }
        const Router& router() const { return mRouter; }
        Expert& expert(const size_t index) { return mExperts[index]; }
        const Expert& expert(const size_t index) const { return mExperts[index]; }

    private:
        // std::exp behind a tiny wrapper so the <cmath> include stays local.
        static double exp_double(double x);

        /**
         * Fill indices[0..K) with the indices of the K largest values,
         * highest first. Ties resolve to the lowest index. O(K * NumExperts),
         * which is fine for the small expert counts MoE targets.
         */
        static void selectTopK(ValueType const* const values, size_t* indices)
        {
            bool used[NumExperts];
            for (size_t e = 0; e < NumExperts; ++e)
            {
                used[e] = false;
            }
            for (size_t j = 0; j < K; ++j)
            {
                size_t best = NumExperts;
                for (size_t e = 0; e < NumExperts; ++e)
                {
                    if (used[e])
                    {
                        continue;
                    }
                    if (best == NumExperts || values[e] > values[best])
                    {
                        best = e;
                    }
                }
                indices[j] = best;
                used[best] = true;
            }
        }

        Router mRouter;
        Expert mExperts[NumExperts];
        size_t mLastIndices[K];
        ValueType mLastGates[K];

        static_assert(NumExperts > 0, "Number of experts must be > 0.");
        static_assert(K > 0, "K must be > 0.");
        static_assert(K <= NumExperts, "K must be <= NumExperts.");
        static_assert(InputDim > 0, "Input dimension must be > 0.");
        static_assert(OutputDim > 0, "Output dimension must be > 0.");
        static_assert(Expert::OutputSize == OutputDim,
                      "Expert OutputSize must equal MoE OutputDim.");
        static_assert(Expert::InputSize == InputDim,
                      "Expert InputSize must equal MoE InputDim.");
    };
#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
}

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
#include <cmath>

namespace tinymind {

    template<typename ValueType, size_t InputDim, size_t OutputDim,
             size_t NumExperts, size_t K,
             template<typename, size_t, size_t> class ExpertType>
    double TopKMixtureOfExperts<ValueType, InputDim, OutputDim, NumExperts, K,
                                ExpertType>::exp_double(double x)
    {
        return std::exp(x);
    }

} // namespace tinymind
#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
