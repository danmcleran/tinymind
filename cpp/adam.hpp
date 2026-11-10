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
#include "include/tinymind_traits.hpp"

#include <cstddef>

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
#include <cmath>
#endif

namespace tinymind {
    /**
     * No-op optimizer policy (default). Weight updates use SGD with momentum.
     */
    template<typename ValueType>
    struct NullOptimizerPolicy
    {
        static const bool IsAdam = false;

        void initialize() {}
        void step() {}
    };

    /**
     * Square root approximation for fixed-point QValue types.
     *
     * FPU-free integer Newton's method on the raw representation.
     * Q-format identity: for value = r / 2^N, sqrt(value) in Q form is
     * isqrt(r << N) — the integer sqrt of the raw value left-shifted by
     * the number of fractional bits. The shift fits in the QValue's
     * MultiplicationResultFullWidthValueType (always 2x the raw width)
     * for any supported Q format. Negative inputs clamp to 0.
     *
     * Convergence: O(log W) iterations where W is the bit width of the
     * intermediate, so ~6 iterations for 64-bit and ~7 for 128-bit.
     */
    template<typename ValueType>
    struct SquareRootApproximation
    {
        typedef typename ValueType::FullWidthValueType                     FullWidthValueType;
        typedef typename ValueType::MultiplicationResultFullWidthValueType WideType;

        static ValueType sqrt(const ValueType& value)
        {
            const FullWidthValueType raw = value.getValue();
            if (raw <= static_cast<FullWidthValueType>(0))
            {
                return ValueType(static_cast<FullWidthValueType>(0));
            }

            const WideType n = static_cast<WideType>(raw) << ValueType::NumberOfFractionalBits;
            WideType x = n;
            WideType y = (x + static_cast<WideType>(1)) / static_cast<WideType>(2);
            while (y < x)
            {
                x = y;
                y = (x + n / x) / static_cast<WideType>(2);
            }
            return ValueType(static_cast<FullWidthValueType>(x));
        }
    };

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
    template<>
    struct SquareRootApproximation<float>
    {
        static float sqrt(const float& value)
        {
            return std::sqrt(value);
        }
    };

    template<>
    struct SquareRootApproximation<double>
    {
        static double sqrt(const double& value)
        {
            return std::sqrt(value);
        }
    };
#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD

    namespace detail
    {
        constexpr unsigned long long isqrtRounded(const unsigned long long n)
        {
            unsigned long long low = 0;
            unsigned long long high = (n < 2ULL) ? n : ((n >> 1) < 4294967295ULL ? (n >> 1) : 4294967295ULL);
            while (low < high)
            {
                const unsigned long long mid = low + (high - low + 1ULL) / 2ULL;
                if (mid <= n / mid)
                {
                    low = mid;
                }
                else
                {
                    high = mid - 1ULL;
                }
            }
            // Round to nearest: (r + 0.5)^2 = r^2 + r + 0.25.
            return ((n - low * low) > low) ? (low + 1ULL) : low;
        }

        // round(sqrt(D * 2^bits)), the raw fraction of sqrt(2^-bits / (1 - beta2))
        // for 1 - beta2 = 1/D, kept inside (0, 1).
        constexpr unsigned long long adamEpsilonLsbs(const unsigned long long d, const unsigned bits)
        {
            const unsigned long long maxLsbs = (bits >= 64u) ? ~0ULL : ((1ULL << bits) - 1ULL);
            unsigned long long lsbs = 0;
            if (bits <= 56u)
            {
                lsbs = isqrtRounded(d << bits);
            }
            else
            {
                // D * 2^bits does not fit; sqrt(D * 2^bits) = sqrt(D * 2^(bits mod 2)) * 2^(bits / 2).
                lsbs = isqrtRounded(d << (bits % 2u)) << (bits / 2u);
            }
            if (lsbs > maxLsbs)
            {
                lsbs = maxLsbs;
            }
            return (lsbs == 0ULL) ? 1ULL : lsbs;
        }
    }

    /**
     * Fixed-point Adam epsilon for 1 - beta2 = 1/D: epsilon = sqrt(LSB / (1 - beta2)),
     * the gradient magnitude below which the second-moment increment
     * (1 - beta2) * g^2 rounds to zero. See AdamOptimizer for why.
     */
    template<typename ValueType, unsigned long long D>
    struct QAdamEpsilonLsbs
    {
        static const unsigned long long value = detail::adamEpsilonLsbs(D, fractional_bits_of<ValueType>::value);
    };

    template<typename ValueType, unsigned long long D>
    const unsigned long long QAdamEpsilonLsbs<ValueType, D>::value;

    /**
     * Adam optimizer policy for use with BackPropagationParent.
     *
     * Adam maintains per-parameter first-moment (mean) and second-moment
     * (uncentered variance) estimates, enabling adaptive learning rates.
     * This is especially valuable for gated networks (LSTM, GRU) where
     * different gates need different effective learning rates.
     *
     * The first and second moments reuse the existing mDeltaWeight and
     * mPreviousDeltaWeight storage in TrainableConnection, so no additional
     * per-connection memory is required beyond what SGD already uses. Bias
     * connections are updated the same way, through updateBiasWeights().
     *
     * The network calls step() once per weight update, before any weight
     * moves (every trainNetwork() with BatchSize 1, once per batch
     * otherwise), so the bias-correction timestep t counts updates, not
     * layers.
     *
     * Each constant is an (IntegerPart, FractionalPart) pair for the
     * QValue(int, fraction) constructor; FractionalPart is a raw count of
     * LSBs in ValueType's own format. The defaults are computed for
     * ValueType's format, each rounded to it and kept inside (0, 1):
     *   beta1 = 0.9, beta2 = 0.99, epsilon = sqrt(LSB / (1 - beta2))
     * In Q16.16 that is beta1 = 58982/65536, beta2 = 64881/65536 and
     * epsilon = 2560/65536 (0.039); in Q32.32 epsilon is 1.5e-4.
     *
     * These are not the floating-point textbook values (beta2 = 0.999,
     * epsilon = 1e-8, which AdamOptimizerFloat uses), because the second
     * moment underflows in fixed point: (1 - beta2) * g^2 rounds to zero
     * for any |g| < sqrt(LSB / (1 - beta2)), so v stops tracking small
     * gradients and the step lr * m / (sqrt(v) + epsilon) is limited only by
     * epsilon. With epsilon at one LSB that step was hundreds of times the
     * textbook one; with epsilon = sqrt(LSB / (1 - beta2)), the gradient
     * size at which v underflows, it stays within about one learning rate.
     * beta2 = 0.99 keeps that epsilon small (0.039 in Q16.16, against 0.12
     * for beta2 = 0.999). On the Q16.16 XOR network in unit_test/nn, the
     * textbook constants reach the error target on 6 of 24 seeds and these
     * defaults on 24 of 24. Pass explicit constants to override, e.g.
     * AdamOptimizer<Q, 0, QFractionLsbs<Q, 9, 10>::value, 0, QFractionLsbs<Q, 999, 1000>::value, 0, QAdamEpsilonLsbs<Q, 1000>::value>.
     * Q8.8 has too few fractional bits for Adam (its default epsilon is
     * 0.625). For floating-point, use AdamOptimizerFloat.
     */
    template<typename ValueType,
             int Beta1Int = 0, unsigned long long Beta1Frac = QFractionLsbs<ValueType, 9, 10>::value,
             int Beta2Int = 0, unsigned long long Beta2Frac = QFractionLsbs<ValueType, 99, 100>::value,
             int EpsilonInt = 0, unsigned long long EpsilonFrac = QAdamEpsilonLsbs<ValueType, 100>::value>
    struct AdamOptimizer
    {
        static const bool IsAdam = true;

        AdamOptimizer() : mTimestep(0), mBeta1Power(one()), mBeta2Power(one())
        {
        }

        void initialize()
        {
            mTimestep = 0;
            mBeta1Power = one();
            mBeta2Power = one();
        }

        void step()
        {
            ++mTimestep;
            // beta^t is tracked incrementally, so the bias correction stays
            // right for every t (it used to stop at t = 100) and costs two
            // multiplies per training step instead of a t-long loop per weight.
            mBeta1Power = mBeta1Power * beta1();
            mBeta2Power = mBeta2Power * beta2();
        }

        size_t getTimestep() const
        {
            return mTimestep;
        }

        static ValueType beta1()
        {
            static const ValueType value(static_cast<typename ValueType::FixedPartFieldType>(Beta1Int), static_cast<typename ValueType::FractionalPartFieldType>(Beta1Frac));
            return value;
        }

        static ValueType beta2()
        {
            static const ValueType value(static_cast<typename ValueType::FixedPartFieldType>(Beta2Int), static_cast<typename ValueType::FractionalPartFieldType>(Beta2Frac));
            return value;
        }

        static ValueType epsilon()
        {
            static const ValueType value(static_cast<typename ValueType::FixedPartFieldType>(EpsilonInt), static_cast<typename ValueType::FractionalPartFieldType>(EpsilonFrac));
            return value;
        }

        template<typename LayerType, typename PreviousLayerType>
        void updateWeights(PreviousLayerType& previousLayer, const size_t neuron, const size_t conn, const ValueType& gradient, const ValueType& learningRate)
        {
            ValueType m = previousLayer.getFirstMomentForNeuronAndConnection(neuron, conn);
            ValueType v = previousLayer.getSecondMomentForNeuronAndConnection(neuron, conn);
            const ValueType update = this->computeUpdate(m, v, gradient, learningRate);

            previousLayer.setFirstMomentForNeuronAndConnection(neuron, conn, m);
            previousLayer.setSecondMomentForNeuronAndConnection(neuron, conn, v);

            const ValueType currentWeight = previousLayer.getWeightForNeuronAndConnection(neuron, conn);
            previousLayer.setWeightForNeuronAndConnection(neuron, conn, currentWeight + update);
        }

        template<typename LayerType, typename PreviousLayerType>
        void updateBiasWeights(PreviousLayerType& previousLayer, const size_t conn, const ValueType& gradient, const ValueType& learningRate)
        {
            ValueType m = previousLayer.getBiasNeuronFirstMomentForConnection(conn);
            ValueType v = previousLayer.getBiasNeuronSecondMomentForConnection(conn);
            const ValueType update = this->computeUpdate(m, v, gradient, learningRate);

            previousLayer.setBiasNeuronFirstMomentForConnection(conn, m);
            previousLayer.setBiasNeuronSecondMomentForConnection(conn, v);

            const ValueType currentWeight = previousLayer.getBiasNeuronWeightForConnection(conn);
            previousLayer.setBiasNeuronWeightForConnection(conn, currentWeight + update);
        }

    private:
        static ValueType one()
        {
            return ValueType(static_cast<typename ValueType::FixedPartFieldType>(1), static_cast<typename ValueType::FractionalPartFieldType>(0));
        }

        // Updates the moments in place and returns lr * mHat / (sqrt(vHat) + epsilon).
        ValueType computeUpdate(ValueType& m, ValueType& v, const ValueType& gradient, const ValueType& learningRate) const
        {
            m = beta1() * m + (one() - beta1()) * gradient;
            v = beta2() * v + (one() - beta2()) * gradient * gradient;

            ValueType mHat = m;
            ValueType vHat = v;
            const ValueType zero(static_cast<typename ValueType::FullWidthValueType>(0));
            const ValueType beta1Correction = one() - mBeta1Power;
            const ValueType beta2Correction = one() - mBeta2Power;

            // Both corrections are at least 1 LSB once step() has run; before
            // the first step (t = 0) they are 0 and the moments are used as is.
            if (beta1Correction > zero)
            {
                mHat = m / beta1Correction;
            }
            if (beta2Correction > zero)
            {
                vHat = v / beta2Correction;
            }

            const ValueType sqrtV = SquareRootApproximation<ValueType>::sqrt(vHat);
            return learningRate * mHat / (sqrtV + epsilon());
        }

        size_t mTimestep;
        ValueType mBeta1Power;
        ValueType mBeta2Power;
    };

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
    /**
     * Floating-point Adam optimizer with standard hyperparameters
     * (beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8). Uses double-precision
     * constants directly instead of QValue constructors. Requires
     * TINYMIND_ENABLE_STD because std::sqrt is needed for moment
     * normalization. The network calls step() once per weight update.
     */
    template<typename ValueType>
    struct AdamOptimizerFloat
    {
        static const bool IsAdam = true;

        AdamOptimizerFloat() : mTimestep(0), mBeta1(0.9), mBeta2(0.999), mEpsilon(1e-8), mBeta1Power(1.0), mBeta2Power(1.0)
        {
        }

        void initialize()
        {
            mTimestep = 0;
            mBeta1Power = 1.0;
            mBeta2Power = 1.0;
        }

        void step()
        {
            ++mTimestep;
            mBeta1Power *= mBeta1;
            mBeta2Power *= mBeta2;
        }

        size_t getTimestep() const
        {
            return mTimestep;
        }

        template<typename LayerType, typename PreviousLayerType>
        void updateWeights(PreviousLayerType& previousLayer, const size_t neuron, const size_t conn, const ValueType& gradient, const ValueType& learningRate)
        {
            ValueType m = previousLayer.getFirstMomentForNeuronAndConnection(neuron, conn);
            ValueType v = previousLayer.getSecondMomentForNeuronAndConnection(neuron, conn);
            const ValueType update = this->computeUpdate(m, v, gradient, learningRate);

            previousLayer.setFirstMomentForNeuronAndConnection(neuron, conn, m);
            previousLayer.setSecondMomentForNeuronAndConnection(neuron, conn, v);

            const ValueType currentWeight = previousLayer.getWeightForNeuronAndConnection(neuron, conn);
            previousLayer.setWeightForNeuronAndConnection(neuron, conn, currentWeight + update);
        }

        template<typename LayerType, typename PreviousLayerType>
        void updateBiasWeights(PreviousLayerType& previousLayer, const size_t conn, const ValueType& gradient, const ValueType& learningRate)
        {
            ValueType m = previousLayer.getBiasNeuronFirstMomentForConnection(conn);
            ValueType v = previousLayer.getBiasNeuronSecondMomentForConnection(conn);
            const ValueType update = this->computeUpdate(m, v, gradient, learningRate);

            previousLayer.setBiasNeuronFirstMomentForConnection(conn, m);
            previousLayer.setBiasNeuronSecondMomentForConnection(conn, v);

            const ValueType currentWeight = previousLayer.getBiasNeuronWeightForConnection(conn);
            previousLayer.setBiasNeuronWeightForConnection(conn, currentWeight + update);
        }

    private:
        ValueType computeUpdate(ValueType& m, ValueType& v, const ValueType& gradient, const ValueType& learningRate) const
        {
            m = static_cast<ValueType>(mBeta1) * m + static_cast<ValueType>(1.0 - mBeta1) * gradient;
            v = static_cast<ValueType>(mBeta2) * v + static_cast<ValueType>(1.0 - mBeta2) * gradient * gradient;

            ValueType mHat = m;
            ValueType vHat = v;
            if (mTimestep > 0)
            {
                mHat = m / static_cast<ValueType>(1.0 - mBeta1Power);
                vHat = v / static_cast<ValueType>(1.0 - mBeta2Power);
            }

            const ValueType sqrtV = static_cast<ValueType>(std::sqrt(static_cast<double>(vHat)));
            return learningRate * mHat / (sqrtV + static_cast<ValueType>(mEpsilon));
        }

        size_t mTimestep;
        double mBeta1;
        double mBeta2;
        double mEpsilon;
        double mBeta1Power;
        double mBeta2Power;
    };
#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD

}
