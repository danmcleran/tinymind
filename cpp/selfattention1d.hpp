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
     * Linear self-attention layer for sequence processing on embedded targets.
     *
     * Uses linear attention (kernel feature map instead of softmax) to avoid
     * the O(N^2) cost of standard attention. The forward pass computes:
     *
     *   Q' = ReLU(X * W_q)           N x ProjectionDim
     *   K' = ReLU(X * W_k)           N x ProjectionDim
     *   V  = X * W_v                 N x ProjectionDim
     *   KV = K'^T * V                ProjectionDim x ProjectionDim
     *   Out = Q' * KV                N x ProjectionDim
     *
     * Complexity is O(N * D * P + N * P^2) instead of O(N^2 * D), where
     * P = ProjectionDim. All dimensions are compile-time template parameters
     * with zero dynamic allocation.
     *
     * Output feeds into any TinyMind network as input:
     *   SelfAttention1D<double, 32, 16, 8> attn;
     *   NeuralNetwork<double, 32*8, ...> mlp;  // flattened attention output
     *
     *   attn.forward(inputSequence, attendedOutput);
     *   mlp.feedForward(attendedOutput);
     *
     * @tparam ValueType      Numeric type (QValue or float/double)
     * @tparam SequenceLength  Number of input time steps (N)
     * @tparam EmbeddingDim    Input feature dimension per time step (D)
     * @tparam ProjectionDim   Dimension of Q, K, V projections (P)
     */
    template<
        typename ValueType,
        size_t SequenceLength,
        size_t EmbeddingDim,
        size_t ProjectionDim>
    class SelfAttention1D
    {
    public:
        static const size_t InputSize = SequenceLength * EmbeddingDim;
        static const size_t OutputSize = SequenceLength * ProjectionDim;
        static const size_t WeightsPerProjection = EmbeddingDim * ProjectionDim;
        static const size_t TotalWeights = 3 * WeightsPerProjection;
        static const size_t BiasesPerProjection = ProjectionDim;
        static const size_t TotalBiases = 3 * BiasesPerProjection;

        SelfAttention1D()
        {
            for (size_t i = 0; i < TotalWeights; ++i)
            {
                mWeights[i] = ValueType(0);
                mGradients[i] = ValueType(0);
            }
            for (size_t i = 0; i < TotalBiases; ++i)
            {
                mBiases[i] = ValueType(0);
                mBiasGradients[i] = ValueType(0);
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
                mGradients[i] = ValueType(0);
            }
            for (size_t i = 0; i < TotalBiases; ++i)
            {
                mBiases[i] = ValueType(0);
                mBiasGradients[i] = ValueType(0);
            }
        }

        /**
         * Forward pass: linear self-attention over the input sequence.
         *
         * @param input  Array of InputSize values, row-major layout:
         *               [t0_d0, t0_d1, ..., t0_dD, t1_d0, ...]
         *               where t = time step, d = embedding dimension
         * @param output Array of OutputSize values, row-major layout:
         *               [t0_p0, t0_p1, ..., t0_pP, t1_p0, ...]
         *               where p = projection dimension
         */
        void forward(ValueType const* const input, ValueType* output)
        {
            // Cache input for backward pass
            for (size_t i = 0; i < InputSize; ++i)
            {
                mInput[i] = input[i];
            }

            // Step 1: Compute Q = X * W_q + b_q, then apply ReLU
            matmul(input, &mWeights[0], &mBiases[0], mQ, SequenceLength, EmbeddingDim, ProjectionDim);
            relu(mQ, SequenceLength * ProjectionDim);

            // Step 2: Compute K = X * W_k + b_k, then apply ReLU
            matmul(input, &mWeights[WeightsPerProjection], &mBiases[BiasesPerProjection], mK, SequenceLength, EmbeddingDim, ProjectionDim);
            relu(mK, SequenceLength * ProjectionDim);

            // Step 3: Compute V = X * W_v + b_v (no activation)
            matmul(input, &mWeights[2 * WeightsPerProjection], &mBiases[2 * BiasesPerProjection], mV, SequenceLength, EmbeddingDim, ProjectionDim);

            // Step 4: Compute KV = K'^T * V  (ProjectionDim x ProjectionDim)
            matmulTransA(mK, mV, mKV, ProjectionDim, SequenceLength, ProjectionDim);

            // Step 5: Compute Out = Q' * KV  (SequenceLength x ProjectionDim)
            matmulNoBias(mQ, mKV, output, SequenceLength, ProjectionDim, ProjectionDim);

            // Cache output for backward pass
            for (size_t i = 0; i < OutputSize; ++i)
            {
                mOutput[i] = output[i];
            }
        }

        /**
         * Compute gradients given output deltas and the input that produced them.
         * Call after forward pass and error computation.
         *
         * @param outputDeltas Array of OutputSize gradient values from the next layer
         */
        void computeGradients(ValueType const* const outputDeltas)
        {
            // Zero all gradients
            for (size_t i = 0; i < TotalWeights; ++i)
            {
                mGradients[i] = ValueType(0);
            }
            for (size_t i = 0; i < TotalBiases; ++i)
            {
                mBiasGradients[i] = ValueType(0);
            }

            // dOut = outputDeltas (SequenceLength x ProjectionDim)

            // Step 5 backward: Out = Q' * KV
            // dQ' = dOut * KV^T  (SequenceLength x ProjectionDim)
            ValueType dQ[SequenceLength * ProjectionDim];
            matmulTransB(outputDeltas, mKV, dQ, SequenceLength, ProjectionDim, ProjectionDim);

            // dKV = Q'^T * dOut  (ProjectionDim x ProjectionDim)
            ValueType dKV[ProjectionDim * ProjectionDim];
            matmulTransA(mQ, outputDeltas, dKV, ProjectionDim, SequenceLength, ProjectionDim);

            // Step 4 backward: KV = K'^T * V
            // dK' = V * dKV^T  (SequenceLength x ProjectionDim)
            ValueType dK[SequenceLength * ProjectionDim];
            matmulTransB(mV, dKV, dK, SequenceLength, ProjectionDim, ProjectionDim);

            // dV = K' * dKV  (SequenceLength x ProjectionDim)
            ValueType dV[SequenceLength * ProjectionDim];
            matmulNoBias(mK, dKV, dV, SequenceLength, ProjectionDim, ProjectionDim);

            // Steps 1-2 backward: apply ReLU derivative to dQ and dK
            reluDerivative(mQ, dQ, SequenceLength * ProjectionDim);
            reluDerivative(mK, dK, SequenceLength * ProjectionDim);

            // Compute weight gradients: dW = X^T * dProjection
            // W_q gradients
            matmulTransA(mInput, dQ, &mGradients[0], EmbeddingDim, SequenceLength, ProjectionDim);
            // W_k gradients
            matmulTransA(mInput, dK, &mGradients[WeightsPerProjection], EmbeddingDim, SequenceLength, ProjectionDim);
            // W_v gradients
            matmulTransA(mInput, dV, &mGradients[2 * WeightsPerProjection], EmbeddingDim, SequenceLength, ProjectionDim);

            // Compute bias gradients: sum deltas across sequence
            computeBiasGradients(dQ, &mBiasGradients[0]);
            computeBiasGradients(dK, &mBiasGradients[BiasesPerProjection]);
            computeBiasGradients(dV, &mBiasGradients[2 * BiasesPerProjection]);
        }

        /**
         * Update weights using SGD with learning rate.
         * @param learningRate Step size for weight update
         */
        void updateWeights(const ValueType& learningRate)
        {
            for (size_t i = 0; i < TotalWeights; ++i)
            {
                mWeights[i] += learningRate * mGradients[i];
            }
            for (size_t i = 0; i < TotalBiases; ++i)
            {
                mBiases[i] += learningRate * mBiasGradients[i];
            }
        }

        // Weight accessors for serialization
        ValueType getWeight(const size_t index) const { return mWeights[index]; }
        void setWeight(const size_t index, const ValueType& value) { mWeights[index] = value; }
        ValueType getGradient(const size_t index) const { return mGradients[index]; }

        ValueType getBias(const size_t index) const { return mBiases[index]; }
        void setBias(const size_t index, const ValueType& value) { mBiases[index] = value; }
        ValueType getBiasGradient(const size_t index) const { return mBiasGradients[index]; }

        /**
         * Get a projection weight by matrix, row, and column.
         * @param projection 0=W_q, 1=W_k, 2=W_v
         * @param row Row index [0, EmbeddingDim)
         * @param col Column index [0, ProjectionDim)
         */
        ValueType getProjectionWeight(const size_t projection, const size_t row, const size_t col) const
        {
            return mWeights[projection * WeightsPerProjection + row * ProjectionDim + col];
        }

        void setProjectionWeight(const size_t projection, const size_t row, const size_t col, const ValueType& value)
        {
            mWeights[projection * WeightsPerProjection + row * ProjectionDim + col] = value;
        }

        /**
         * Get a projection bias.
         * @param projection 0=W_q, 1=W_k, 2=W_v
         * @param index Bias index [0, ProjectionDim)
         */
        ValueType getProjectionBias(const size_t projection, const size_t index) const
        {
            return mBiases[projection * BiasesPerProjection + index];
        }

        void setProjectionBias(const size_t projection, const size_t index, const ValueType& value)
        {
            mBiases[projection * BiasesPerProjection + index] = value;
        }

    private:
        /**
         * Matrix multiply: C = A * B + bias (row-major)
         * A is (M x K), B is (K x N), C is (M x N), bias is (N)
         */
        static void matmul(ValueType const* A, ValueType const* B, ValueType const* bias,
                           ValueType* C, const size_t M, const size_t K, const size_t N)
        {
            for (size_t m = 0; m < M; ++m)
            {
                for (size_t n = 0; n < N; ++n)
                {
                    ValueType sum = bias[n];
                    for (size_t k = 0; k < K; ++k)
                    {
                        sum += A[m * K + k] * B[k * N + n];
                    }
                    C[m * N + n] = sum;
                }
            }
        }

        /**
         * Matrix multiply without bias: C = A * B (row-major)
         */
        static void matmulNoBias(ValueType const* A, ValueType const* B,
                                 ValueType* C, const size_t M, const size_t K, const size_t N)
        {
            for (size_t m = 0; m < M; ++m)
            {
                for (size_t n = 0; n < N; ++n)
                {
                    ValueType sum = ValueType(0);
                    for (size_t k = 0; k < K; ++k)
                    {
                        sum += A[m * K + k] * B[k * N + n];
                    }
                    C[m * N + n] = sum;
                }
            }
        }

        /**
         * Matrix multiply with A transposed: C = A^T * B (row-major)
         * A is (K x M) so A^T is (M x K), B is (K x N), C is (M x N)
         */
        static void matmulTransA(ValueType const* A, ValueType const* B,
                                 ValueType* C, const size_t M, const size_t K, const size_t N)
        {
            for (size_t m = 0; m < M; ++m)
            {
                for (size_t n = 0; n < N; ++n)
                {
                    ValueType sum = ValueType(0);
                    for (size_t k = 0; k < K; ++k)
                    {
                        sum += A[k * M + m] * B[k * N + n];
                    }
                    C[m * N + n] = sum;
                }
            }
        }

        /**
         * Matrix multiply with B transposed: C = A * B^T (row-major)
         * A is (M x K), B is (N x K) so B^T is (K x N), C is (M x N)
         */
        static void matmulTransB(ValueType const* A, ValueType const* B,
                                 ValueType* C, const size_t M, const size_t K, const size_t N)
        {
            for (size_t m = 0; m < M; ++m)
            {
                for (size_t n = 0; n < N; ++n)
                {
                    ValueType sum = ValueType(0);
                    for (size_t k = 0; k < K; ++k)
                    {
                        sum += A[m * K + k] * B[n * K + k];
                    }
                    C[m * N + n] = sum;
                }
            }
        }

        /**
         * In-place ReLU activation.
         */
        static void relu(ValueType* data, const size_t count)
        {
            for (size_t i = 0; i < count; ++i)
            {
                if (data[i] < ValueType(0))
                {
                    data[i] = ValueType(0);
                }
            }
        }

        /**
         * Apply ReLU derivative: zero out gradient where activation was zero.
         */
        static void reluDerivative(ValueType const* activation, ValueType* gradient, const size_t count)
        {
            for (size_t i = 0; i < count; ++i)
            {
                if (activation[i] <= ValueType(0))
                {
                    gradient[i] = ValueType(0);
                }
            }
        }

        /**
         * Sum deltas across the sequence dimension to get bias gradients.
         */
        static void computeBiasGradients(ValueType const* deltas, ValueType* biasGrad)
        {
            for (size_t p = 0; p < ProjectionDim; ++p)
            {
                biasGrad[p] = ValueType(0);
            }
            for (size_t t = 0; t < SequenceLength; ++t)
            {
                for (size_t p = 0; p < ProjectionDim; ++p)
                {
                    biasGrad[p] += deltas[t * ProjectionDim + p];
                }
            }
        }

        // Projection weights: W_q, W_k, W_v stored contiguously
        ValueType mWeights[TotalWeights];
        ValueType mGradients[TotalWeights];

        // Projection biases: b_q, b_k, b_v stored contiguously
        ValueType mBiases[TotalBiases];
        ValueType mBiasGradients[TotalBiases];

        // Cached intermediate values for backward pass
        ValueType mInput[InputSize];
        ValueType mQ[SequenceLength * ProjectionDim];
        ValueType mK[SequenceLength * ProjectionDim];
        ValueType mV[SequenceLength * ProjectionDim];
        ValueType mKV[ProjectionDim * ProjectionDim];
        ValueType mOutput[OutputSize];

        static_assert(SequenceLength > 0, "Sequence length must be > 0.");
        static_assert(EmbeddingDim > 0, "Embedding dimension must be > 0.");
        static_assert(ProjectionDim > 0, "Projection dimension must be > 0.");
    };
}
