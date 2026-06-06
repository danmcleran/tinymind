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

// Q-Format value type. Q16.16: KAN B-spline training needs more range and
// precision than Q8.8 provides (Q8.8 saturates and fails to converge). The
// learning-rate constants below are defined relative to NumberOfFractionalBits,
// so the effective rates are unchanged -- only the numeric headroom grows.
static const size_t NUMBER_OF_FIXED_BITS = 16;
static const size_t NUMBER_OF_FRACTIONAL_BITS = 16;
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
        // Symmetric small init in [-0.5, 0.5] (raw Q-format units). Small,
        // symmetric weights keep the KAN B-spline training stable -- the old
        // formula produced an asymmetric [-1, 2) range that diverged.
        const FullWidthValueType half =
            tinymind::Constants<ValueType>::one().getValue() / 2;
        const FullWidthValueType weight =
            (static_cast<FullWidthValueType>(rand()) % (2 * half + 1)) - half;
        return ValueType(weight);
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
        // ~0.03125 (lower than the old 0.0625 for stable convergence).
        static const ValueType rate(0, (1 << (ValueType::NumberOfFractionalBits - 5)));
        return rate;
    }

    static ValueType initialMomentumRate()
    {
        // ~0.0625 momentum (the old 0.25 was too high and drove divergence).
        static const ValueType rate(0, (1 << (ValueType::NumberOfFractionalBits - 4)));
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
