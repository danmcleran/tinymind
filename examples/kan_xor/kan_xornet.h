/**
* Copyright (c) 2025 Dan McLeran
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
#include "kan.hpp"

// Q-Format value type
static const size_t NUMBER_OF_FIXED_BITS = 8;
static const size_t NUMBER_OF_FRACTIONAL_BITS = 8;
typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true> ValueType;
typedef typename ValueType::FullWidthValueType FullWidthValueType;

// KAN architecture
static const size_t NUMBER_OF_INPUTS = 2;
static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 5;
static const size_t NUMBER_OF_OUTPUTS = 1;
static const size_t GRID_SIZE = 5;
static const size_t SPLINE_DEGREE = 1; // Piecewise linear - best for fixed-point

// Random number generator
struct RandomNumberGenerator
{
    static ValueType generateRandomWeight()
    {
        // Generate a random number between -1..1 in the Q Format full width type
        const FullWidthValueType weight = (rand() %
                                                    (tinymind::Constants<ValueType>::one().getValue() +
                                                     tinymind::Constants<ValueType>::one().getValue() -
                                                     tinymind::Constants<ValueType>::negativeOne().getValue())) +
                                                     tinymind::Constants<ValueType>::negativeOne().getValue();

        return weight;
    }
};

// KAN-specific network initializer with smaller learning rate
// KAN has more parameters per connection, so needs a lower learning rate
struct KanNetworkInitializer
{
    static ValueType initialAccelerationRate()
    {
        // Very small acceleration rate
        static const ValueType rate(0, (1 << (ValueType::NumberOfFractionalBits - 6)));
        return rate;
    }

    static ValueType initialBiasOutputValue()
    {
        return tinymind::Constants<ValueType>::one();
    }

    static ValueType initialDeltaWeight()
    {
        return tinymind::Constants<ValueType>::zero();
    }

    static ValueType initialGradientValue()
    {
        return tinymind::Constants<ValueType>::zero();
    }

    static ValueType initialLearningRate()
    {
        // Lower learning rate for KAN: ~0.0625 in Q8.8
        static const ValueType rate(0, (1 << (ValueType::NumberOfFractionalBits - 4)));
        return rate;
    }

    static ValueType initialMomentumRate()
    {
        // Moderate momentum
        static const ValueType rate(0, (1 << (ValueType::NumberOfFractionalBits - 2)));
        return rate;
    }

    static ValueType initialOutputValue()
    {
        return tinymind::Constants<ValueType>::zero();
    }

    static ValueType noOpDeltaWeight()
    {
        return tinymind::Constants<ValueType>::one();
    }

    static ValueType noOpWeight()
    {
        return tinymind::Constants<ValueType>::one();
    }
};

// Typedef of transfer functions for the KAN
typedef tinymind::KanTransferFunctions<ValueType,
                                       RandomNumberGenerator,
                                       NUMBER_OF_OUTPUTS,
                                       KanNetworkInitializer> TransferFunctionsType;

// Typedef the KAN network itself
typedef tinymind::KolmogorovArnoldNetwork<ValueType,
                                          NUMBER_OF_INPUTS,
                                          NUMBER_OF_HIDDEN_LAYERS,
                                          NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                          NUMBER_OF_OUTPUTS,
                                          TransferFunctionsType,
                                          true,           // IsTrainable
                                          1,              // BatchSize
                                          GRID_SIZE,
                                          SPLINE_DEGREE> KanNetworkType;
