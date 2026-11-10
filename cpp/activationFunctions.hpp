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
#include "include/tinymind_platform.hpp"
#include "include/tinymind_traits.hpp"

#include <cstddef>
#include <cstdint>

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
#include <cmath>
#endif

#include "lookupTable.hpp"
#include "constants.hpp"
#include "sigmoid.hpp"
#include "tanh.hpp"
#include "exp.hpp"
#include "typeChooser.hpp"

namespace tinymind {
    /**
     * Which argument an activation policy's activationFunctionDerivative()
     * expects. The network stores each neuron's output y = f(x), and by
     * default passes y: sigmoid (y(1-y)), tanh (1-y^2), ReLU and ELU all have
     * derivatives that are functions of their output. GELU's derivative is not
     * (GELU is not one-to-one below zero), so its policy declares
     *
     *     typedef void DerivativeTakesPreActivationTag;
     *
     * and the layers that apply it keep the pre-activation x and pass that
     * instead. Feed-forward and Elman hidden layers and the feed-forward output
     * layer support such policies; the gated (GRU/LSTM) layers reject them at
     * compile time.
     */
    namespace detail
    {
        template<typename T>
        struct ActivationVoidType
        {
            typedef void type;
        };
    }

    template<typename ActivationPolicy, typename Enable = void>
    struct ActivationDerivativeTakesPreActivation
    {
        static const bool value = false;
    };

    template<typename ActivationPolicy>
    struct ActivationDerivativeTakesPreActivation<ActivationPolicy, typename detail::ActivationVoidType<typename ActivationPolicy::DerivativeTakesPreActivationTag>::type>
    {
        static const bool value = true;
    };

    // The same question asked of a transfer-functions policy: false when it does
    // not expose HiddenNeuronActivationPolicy / OutputNeuronActivationPolicy.
    template<typename TransferFunctionsPolicy, typename Enable = void>
    struct HiddenActivationDerivativeTakesPreActivation
    {
        static const bool value = false;
    };

    template<typename TransferFunctionsPolicy>
    struct HiddenActivationDerivativeTakesPreActivation<TransferFunctionsPolicy, typename detail::ActivationVoidType<typename TransferFunctionsPolicy::HiddenNeuronActivationPolicy::DerivativeTakesPreActivationTag>::type>
    {
        static const bool value = true;
    };

    template<typename TransferFunctionsPolicy, typename Enable = void>
    struct OutputActivationDerivativeTakesPreActivation
    {
        static const bool value = false;
    };

    template<typename TransferFunctionsPolicy>
    struct OutputActivationDerivativeTakesPreActivation<TransferFunctionsPolicy, typename detail::ActivationVoidType<typename TransferFunctionsPolicy::OutputNeuronActivationPolicy::DerivativeTakesPreActivationTag>::type>
    {
        static const bool value = true;
    };

    namespace detail
    {
        // A whole number as a ValueType. QValue(int) treats its argument as the
        // raw fixed-point field, so QValue types go through the (fixed,
        // fractional) constructor; floating-point types convert directly.
        template<typename ValueType, bool IsFloat = tinymind::is_floating_point<ValueType>::value>
        struct ActivationWholeNumber
        {
            static ValueType make(const int value)
            {
                return ValueType(static_cast<typename ValueType::FixedPartFieldType>(value), 0u);
            }
        };

        template<typename ValueType>
        struct ActivationWholeNumber<ValueType, true>
        {
            static ValueType make(const int value)
            {
                return static_cast<ValueType>(value);
            }
        };

        // exp(x) and sigmoid(x): lookup tables for QValue types, <cmath> for
        // floating-point types (which needs TINYMIND_ENABLE_FLOAT and _STD).
        template<typename ValueType, bool IsFloat = tinymind::is_floating_point<ValueType>::value>
        struct ActivationExp
        {
            typedef typename ValueType::FullWidthFieldType FullWidthFieldType;
            typedef LookupTable<ValueType> LookupTableType;
            typedef typename ExpValuesTableSelector<ValueType::NumberOfFixedBits, ValueType::NumberOfFractionalBits, ValueType::IsSigned>::ExpTableType ExpTableType;

            static ValueType exp(const ValueType& value)
            {
                static const ptrdiff_t MAX_ACTIVATION_INDEX = (((sizeof(FullWidthFieldType) * NUMBER_OF_ACTIVATION_TABLE_VALUES) / sizeof(expActivationTable.values[0])) - 1);

                return LookupTableType::getValue(value, &expActivationTable.values[0], MAX_ACTIVATION_INDEX);
            }
        private:
            static const ExpTableType expActivationTable;
            static_assert(ValueType::IsSigned, "Exp activation tables require a signed type.");
        };

        template<typename ValueType, bool IsFloat>
        const typename ExpValuesTableSelector<ValueType::NumberOfFixedBits, ValueType::NumberOfFractionalBits, ValueType::IsSigned>::ExpTableType ActivationExp<ValueType, IsFloat>::expActivationTable;

        template<typename ValueType, bool IsFloat = tinymind::is_floating_point<ValueType>::value>
        struct ActivationSigmoid
        {
            typedef LookupTable<ValueType> LookupTableType;
            typedef typename SigmoidValuesTableSelector<ValueType::NumberOfFixedBits, ValueType::NumberOfFractionalBits, ValueType::IsSigned>::SigmoidTableType SigmoidTableType;

            static ValueType sigmoid(const ValueType& value)
            {
                static const ptrdiff_t MAX_ACTIVATION_INDEX = ((sizeof(sigmoidActivationTable.values) / sizeof(sigmoidActivationTable.values[0])) - 1);

                return LookupTableType::getValue(value, &sigmoidActivationTable.values[0], MAX_ACTIVATION_INDEX);
            }
        private:
            static const SigmoidTableType sigmoidActivationTable;
            static_assert(ValueType::IsSigned, "Sigmoid activation tables require a signed type.");
        };

        template<typename ValueType, bool IsFloat>
        const typename SigmoidValuesTableSelector<ValueType::NumberOfFixedBits, ValueType::NumberOfFractionalBits, ValueType::IsSigned>::SigmoidTableType ActivationSigmoid<ValueType, IsFloat>::sigmoidActivationTable;

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
        template<typename ValueType>
        struct ActivationExp<ValueType, true>
        {
            static ValueType exp(const ValueType& value)
            {
                return std::exp(value);
            }
        };

        template<typename ValueType>
        struct ActivationSigmoid<ValueType, true>
        {
            static ValueType sigmoid(const ValueType& value)
            {
                return static_cast<ValueType>(1) / (static_cast<ValueType>(1) + std::exp(-value));
            }
        };
#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
    }

    template<typename ValueType>
    struct NullActivationPolicy
    {
        static ValueType activationFunction(const ValueType& value)
        {
            (void)value; // suppress unused parameter warning
            return 0;
        }

        static ValueType activationFunctionDerivative(const ValueType& value)
        {
            (void)value; // suppress unused parameter warning
            return 0;
        }
    };

    template<typename ValueType>
    struct LinearActivationPolicy
    {
        static ValueType activationFunction(const ValueType& value)
        {
            return value;
        }

        static ValueType activationFunctionDerivative(const ValueType& value)
        {
            (void)value; // suppress unused parameter warning
            // Must be a real 1.0, not the raw field value 1: the single-arg
            // QValue ctor treats its argument as the raw fixed-point field, so
            // a bare `return 1` yields 2^-frac (~0.0000153 in Q16.16), which
            // starves every weight update flowing through a linear output layer.
            return Constants<ValueType>::one();
        }
    };

    template<typename ValueType>
    struct ReluActivationPolicy
    {
        static ValueType activationFunction(const ValueType& value)
        {
            if(value <= 0)
            {
                return 0;
            }

            return value;
        }

        static ValueType activationFunctionDerivative(const ValueType& value)
        {
            if(value <= 0)
            {
                return 0;
            }

            return Constants<ValueType>::one();
        }
    };

    /**
     * ReLU capped at a whole number: f(x) = min(max(x, 0), MaxValue), e.g.
     * MaxValue = 6 for ReLU6. MaxValue is the value itself (6 means 6.0), for
     * QValue and floating-point types alike, not a raw fixed-point field.
     * The derivative, taken at the output, is 1 strictly between 0 and the
     * cap and 0 at or beyond either end.
     */
    template<typename ValueType, int MaxValue>
    struct CappedReluActivationPolicy
    {
        static const int MAX_VALUE = MaxValue;

        static ValueType cap()
        {
            return detail::ActivationWholeNumber<ValueType>::make(MaxValue);
        }

        static ValueType activationFunction(const ValueType& value)
        {
            const ValueType zeroValue = detail::ActivationWholeNumber<ValueType>::make(0);
            if(!(zeroValue < value))
            {
                return zeroValue;
            }

            const ValueType capValue = cap();
            if(capValue < value)
            {
                return capValue;
            }

            return value;
        }

        static ValueType activationFunctionDerivative(const ValueType& value)
        {
            const ValueType zeroValue = detail::ActivationWholeNumber<ValueType>::make(0);
            if((zeroValue < value) && (value < cap()))
            {
                return detail::ActivationWholeNumber<ValueType>::make(1);
            }

            return zeroValue;
        }
    private:
        static_assert(MaxValue > 0, "Capped Relu needs a positive cap.");
    };

    template<typename ValueType>
    struct SigmoidActivationPolicy
    {
        typedef LookupTable<ValueType> LookupTableType;
        typedef typename SigmoidValuesTableSelector<ValueType::NumberOfFixedBits, ValueType::NumberOfFractionalBits, ValueType::IsSigned>::SigmoidTableType SigmoidTableType;

        static ValueType activationFunction(const ValueType& value)
        {
            static const ptrdiff_t MAX_ACTIVATION_INDEX = ((sizeof(sigmoidActivationTable.values) / sizeof(sigmoidActivationTable.values[0])) - 1);

            const ValueType result = LookupTableType::getValue(value, &sigmoidActivationTable.values[0], MAX_ACTIVATION_INDEX);

            return result;
        }

        static ValueType activationFunctionDerivative(const ValueType& value)
        {
            //Approximation for 1st derivative of sigmoid
            const ValueType oneMinusValue = (Constants<ValueType>::one() - value);

            return (value * oneMinusValue);
        }
    private:
        static const SigmoidTableType sigmoidActivationTable;
        static_assert(ValueType::IsSigned, "Sigmoid activation tables require a signed type.");
    };

    template<typename ValueType>
    const typename SigmoidValuesTableSelector<ValueType::NumberOfFixedBits, ValueType::NumberOfFractionalBits, ValueType::IsSigned>::SigmoidTableType SigmoidActivationPolicy<ValueType>::sigmoidActivationTable;

    template<typename ValueType>
    struct TanhActivationPolicy
    {
        typedef typename ValueType::FullWidthFieldType FullWidthFieldType;
        typedef LookupTable<ValueType> LookupTableType;
        typedef typename TanhValuesTableSelector<ValueType::NumberOfFixedBits, ValueType::NumberOfFractionalBits, ValueType::IsSigned>::TanhTableType TanhTableType;

        static ValueType activationFunction(const ValueType& value)
        {
            static const ptrdiff_t MAX_ACTIVATION_INDEX = (((sizeof(FullWidthFieldType) * NUMBER_OF_ACTIVATION_TABLE_VALUES) / sizeof(tanhActivationTable.values[0])) - 1);

            const ValueType result = LookupTableType::getValue(value, &tanhActivationTable.values[0], MAX_ACTIVATION_INDEX);

            return result;
        }

        static ValueType activationFunctionDerivative(const ValueType& value)
        {
            //Approximation for 1st derivative of tanh
            const ValueType valueSquared = (value * value);

            return (Constants<ValueType>::one() - valueSquared);
        }
    private:
        static const TanhTableType tanhActivationTable;
        static_assert(ValueType::IsSigned, "Tanh activation tables require a signed type.");
    };

    template<typename ValueType>
    const typename TanhValuesTableSelector<ValueType::NumberOfFixedBits, ValueType::NumberOfFractionalBits, ValueType::IsSigned>::TanhTableType TanhActivationPolicy<ValueType>::tanhActivationTable;

    /**
     * Softmax output for ClassifierOutputLayerConfiguration, trained with
     * cross-entropy loss L = -sum(t * log(y)).
     *
     * For softmax followed by cross-entropy, dL/dz = y - t at the logits z.
     * The network's node deltas follow the convention delta = -dL/dz (the
     * feed-forward output layer uses (t - y) * f'(y)) and weights move by
     * +learningRate * delta * input, so activationFunctionDerivative() returns
     * t - y. Targets need not be one-hot; they should sum to 1.
     *
     * QValue types use the exp lookup table; float/double use std::exp
     * (TINYMIND_ENABLE_FLOAT and TINYMIND_ENABLE_STD).
     */
    template<typename ValueType>
    struct SoftmaxActivationPolicy
    {
        static void activationFunction(ValueType const* const values, ValueType* results, const size_t numberOfNerons)
        {
            const ValueType zeroValue = detail::ActivationWholeNumber<ValueType>::make(0);
            ValueType result;
            ValueType sum = zeroValue;

            for(size_t neuron = 0;neuron < numberOfNerons;++neuron)
            {
                result = detail::ActivationExp<ValueType>::exp(values[neuron]);
                results[neuron] = result;
                sum += result;
            }

            if (sum != zeroValue)
            {
                for(size_t neuron = 0;neuron < numberOfNerons;++neuron)
                {
                    results[neuron] /= sum;
                }
            }
        }

        static void activationFunctionDerivative(ValueType const* const values, ValueType const* const targetValues, ValueType* results, const size_t numberOfNerons)
        {
            for(size_t neuron = 0;neuron < numberOfNerons;++neuron)
            {
                results[neuron] = (targetValues[neuron] - values[neuron]);
            }
        }
    };

    /**
     * ELU (Exponential Linear Unit) activation function.
     * f(x) = x              if x > 0
     * f(x) = exp(x) - 1     if x <= 0
     *
     * Avoids the "dying ReLU" problem with smooth negative outputs.
     * Uses the existing exp lookup table for fixed-point.
     */
    template<typename ValueType>
    struct EluActivationPolicy
    {
        typedef typename ValueType::FullWidthFieldType FullWidthFieldType;
        typedef LookupTable<ValueType> LookupTableType;
        typedef typename ExpValuesTableSelector<ValueType::NumberOfFixedBits, ValueType::NumberOfFractionalBits, ValueType::IsSigned>::ExpTableType ExpTableType;

        static ValueType activationFunction(const ValueType& value)
        {
            if (value > Constants<ValueType>::zero())
            {
                return value;
            }

            static const ptrdiff_t MAX_ACTIVATION_INDEX = (((sizeof(FullWidthFieldType) * NUMBER_OF_ACTIVATION_TABLE_VALUES) / sizeof(expActivationTable.values[0])) - 1);

            const ValueType expValue = LookupTableType::getValue(value, &expActivationTable.values[0], MAX_ACTIVATION_INDEX);
            return expValue - Constants<ValueType>::one();
        }

        static ValueType activationFunctionDerivative(const ValueType& value)
        {
            if (value > Constants<ValueType>::zero())
            {
                return Constants<ValueType>::one();
            }

            // For x <= 0: f'(x) = f(x) + 1 = exp(x)
            return value + Constants<ValueType>::one();
        }
    private:
        static const ExpTableType expActivationTable;
        static_assert(ValueType::IsSigned, "ELU activation requires a signed type.");
    };

    template<typename ValueType>
    const typename ExpValuesTableSelector<ValueType::NumberOfFixedBits, ValueType::NumberOfFractionalBits, ValueType::IsSigned>::ExpTableType EluActivationPolicy<ValueType>::expActivationTable;

    namespace detail
    {
        // GELU's 1.702: (1, 0.702 * 2^FractionalBits) for QValue (Q8.8:
        // 1 + 179/256 ~ 1.699; Q16.16: 1 + 46006/65536 ~ 1.702), exact for
        // floating-point types.
        template<typename ValueType, bool IsFloat = tinymind::is_floating_point<ValueType>::value>
        struct GeluScale
        {
            static ValueType value()
            {
                const unsigned frac = static_cast<unsigned>(
                    (702ULL * (1ULL << ValueType::NumberOfFractionalBits)) / 1000ULL);
                return ValueType(1, frac);
            }
        };

        template<typename ValueType>
        struct GeluScale<ValueType, true>
        {
            static ValueType value()
            {
                return static_cast<ValueType>(1.702);
            }
        };
    }

    /**
     * GELU (Gaussian Error Linear Unit) activation function.
     * Uses the sigmoid approximation: GELU(x) ~ x * sigmoid(1.702 * x)
     *
     * QValue types reuse the sigmoid lookup table; float/double use std::exp
     * (TINYMIND_ENABLE_FLOAT and TINYMIND_ENABLE_STD).
     *
     * The derivative is a function of the pre-activation x, not of the output:
     *   f'(x) = sigmoid(1.702*x) + 1.702*x*sigmoid(1.702*x)*(1 - sigmoid(1.702*x))
     * GELU is not one-to-one below zero, so x cannot be recovered from f(x).
     * The policy therefore declares DerivativeTakesPreActivationTag, and the
     * layers that apply it store each neuron's pre-activation and pass it to
     * activationFunctionDerivative() (see ActivationDerivativeTakesPreActivation).
     */
    template<typename ValueType>
    struct GeluActivationPolicy
    {
        typedef void DerivativeTakesPreActivationTag;

        static ValueType activationFunction(const ValueType& value)
        {
            const ValueType scaled = detail::GeluScale<ValueType>::value() * value;
            const ValueType sig = detail::ActivationSigmoid<ValueType>::sigmoid(scaled);

            return value * sig;
        }

        // value is the pre-activation x.
        static ValueType activationFunctionDerivative(const ValueType& value)
        {
            const ValueType scaled = detail::GeluScale<ValueType>::value() * value;
            const ValueType sig = detail::ActivationSigmoid<ValueType>::sigmoid(scaled);
            const ValueType oneValue = detail::ActivationWholeNumber<ValueType>::make(1);

            // f'(x) = sig(s*x) + s*x * sig(s*x) * (1 - sig(s*x))
            return sig + scaled * sig * (oneValue - sig);
        }
    };
}