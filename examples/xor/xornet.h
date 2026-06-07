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
#include <cstdint>
#include <new>

#include "qformat.hpp"
#include "activationFunctions.hpp"
#include "fixedPointTransferFunctions.hpp"
#include "neuralnet.hpp"

// Q-Format value type. Q16.16 (not Q8.8): the coarse 1/256 grid of Q8.8 plus a
// minimal 3-neuron hidden layer made the learning curve lurch through plateaus
// before snapping to a solution. Q16.16 with a slightly wider hidden layer
// gives smooth, monotonic convergence.
static const size_t NUMBER_OF_FIXED_BITS = 16;
static const size_t NUMBER_OF_FRACTIONAL_BITS = 16;
typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true> ValueType;
// typedef the underlying full-width representation type
typedef typename ValueType::FullWidthValueType FullWidthValueType;
// Neural network architecture
static const size_t NUMBER_OF_INPUTS = 2;
static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 4;
static const size_t NUMBER_OF_OUTPUTS = 1;

// Random number generator
struct RandomNumberGenerator
{
    static ValueType generateRandomWeight()
    {
        // Symmetric small init in [-0.5, 0.5] (raw Q-format units). The old
        // formula produced an asymmetric [-1, 2) range that biased the start
        // and made training oscillate before converging.
        const FullWidthValueType half =
            tinymind::Constants<ValueType>::one().getValue() / 2;
        const FullWidthValueType weight =
            (static_cast<FullWidthValueType>(rand()) % (2 * half + 1)) - half;
        return ValueType(weight);
    }
};

// Typedef of transfer functions for the fixed-point neural network
// tanh hidden + sigmoid output: the canonical, stable XOR MLP. (ReLU hidden on
// a 3-neuron XOR in Q8.8 oscillated and failed to converge.)
typedef tinymind::FixedPointTransferFunctions<  ValueType,
                                                RandomNumberGenerator,
                                                tinymind::TanhActivationPolicy<ValueType>,
                                                tinymind::SigmoidActivationPolicy<ValueType>> TransferFunctionsType;

// typedef the neural network itself
typedef tinymind::MultilayerPerceptron< ValueType,
                                        NUMBER_OF_INPUTS,
                                        NUMBER_OF_HIDDEN_LAYERS,
                                        NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                        NUMBER_OF_OUTPUTS,
                                        TransferFunctionsType> NeuralNetworkType;
