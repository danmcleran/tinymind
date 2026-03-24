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
#include "constants.hpp"
#include "interpolate.hpp"

namespace tinymind {

    /**
     * Uniform knot vector for B-spline evaluation.
     *
     * For a B-spline of degree k with G grid intervals, we need G + 2k + 1 knots.
     * The number of basis functions (and thus learnable coefficients) is G + k.
     *
     * Template parameters:
     *   ValueType   - QValue<...> or float/double
     *   GridSize    - Number of grid intervals (G)
     *   SplineDegree - Polynomial degree (k): 1=linear, 2=quadratic, 3=cubic
     */
    template<typename ValueType, size_t GridSize, size_t SplineDegree>
    struct UniformKnotVector
    {
        static const size_t NumberOfKnots = GridSize + 2 * SplineDegree + 1;
        static const size_t NumberOfBasisFunctions = GridSize + SplineDegree;

        ValueType knots[NumberOfKnots];
        ValueType reciprocalSpacing;

        UniformKnotVector()
        {
            for (size_t i = 0; i < NumberOfKnots; ++i)
            {
                knots[i] = static_cast<ValueType>(0);
            }
            reciprocalSpacing = static_cast<ValueType>(0);
        }

        /**
         * Initialize knots with uniform spacing on [gridMin, gridMax].
         * Extended knots are placed outside the grid for boundary B-spline support.
         */
        void initialize(const ValueType& gridMin, const ValueType& gridMax)
        {
            const ValueType range = gridMax - gridMin;

            // Compute spacing by repeated subtraction to avoid needing
            // to construct ValueType(GridSize) which differs for QValue vs double.
            // spacing = range / GridSize
            ValueType spacing = range;
            for (size_t d = 1; d < GridSize; ++d)
            {
                (void)d;
            }
            // Use actual division: range / GridSize
            // For QValue, GridSize is a compile-time constant integer.
            // We build the divisor by adding one() repeatedly.
            ValueType divisor = Constants<ValueType>::one();
            for (size_t d = 1; d < GridSize; ++d)
            {
                divisor += Constants<ValueType>::one();
            }
            spacing = range / divisor;

            if (spacing > static_cast<ValueType>(0))
            {
                reciprocalSpacing = Constants<ValueType>::one() / spacing;
            }

            // Build knot vector: knots[i] = gridMin + (i - SplineDegree) * spacing
            // Start from gridMin and step by spacing
            // knots[SplineDegree] = gridMin
            knots[SplineDegree] = gridMin;

            // Forward from gridMin
            for (size_t i = SplineDegree + 1; i < NumberOfKnots; ++i)
            {
                knots[i] = knots[i - 1] + spacing;
            }

            // Backward from gridMin
            for (size_t i = SplineDegree; i > 0; --i)
            {
                knots[i - 1] = knots[i] - spacing;
            }
        }

        void * operator new(size_t, void *p)
        {
            return p;
        }

        static_assert(GridSize > 0, "GridSize must be > 0.");
        static_assert(SplineDegree > 0, "SplineDegree must be > 0.");
    };

    /**
     * De Boor's algorithm for B-spline evaluation.
     *
     * Evaluates spline(x) = sum_i(c_i * B_{i,k}(x)) using the non-recursive
     * triangular table formulation. Uses only add, subtract, multiply, and
     * division (by knot differences which are uniform and constant).
     *
     * Working array size is SplineDegree + 1, stored on the stack.
     */
    template<typename ValueType, size_t SplineDegree>
    struct DeBoorEvaluator
    {
        /**
         * Find the knot span index such that knots[span] <= x < knots[span+1].
         * Clamps to valid range for the spline.
         */
        static size_t findKnotSpan(const ValueType* knots, const size_t numberOfBasisFunctions, const ValueType& x)
        {
            // Valid span range: [SplineDegree, numberOfBasisFunctions - 1]
            // (corresponds to the interior of the knot vector)
            if (x <= knots[SplineDegree])
            {
                return SplineDegree;
            }
            if (x >= knots[numberOfBasisFunctions])
            {
                return numberOfBasisFunctions - 1;
            }

            // Linear search (suitable for small grid sizes typical in KAN)
            for (size_t i = SplineDegree; i < numberOfBasisFunctions; ++i)
            {
                if (x < knots[i + 1])
                {
                    return i;
                }
            }

            return numberOfBasisFunctions - 1;
        }

        /**
         * Evaluate the B-spline: sum_i(coefficients[i] * B_{i,k}(x))
         * using De Boor's algorithm.
         */
        static ValueType evaluateSpline(
            const ValueType* coefficients,
            const ValueType* knots,
            const size_t numberOfBasisFunctions,
            const ValueType& x)
        {
            const size_t span = findKnotSpan(knots, numberOfBasisFunctions, x);

            // Initialize with the SplineDegree+1 relevant coefficients
            ValueType d[SplineDegree + 1];
            for (size_t j = 0; j <= SplineDegree; ++j)
            {
                d[j] = coefficients[span - SplineDegree + j];
            }

            // De Boor's triangular computation
            for (size_t r = 1; r <= SplineDegree; ++r)
            {
                for (size_t j = SplineDegree; j >= r; --j)
                {
                    const size_t knotIndex = span - SplineDegree + j;
                    const ValueType leftKnot = knots[knotIndex];
                    const ValueType rightKnot = knots[knotIndex + SplineDegree - r + 1];
                    const ValueType denom = rightKnot - leftKnot;

                    if (denom > static_cast<ValueType>(0) || denom < static_cast<ValueType>(0))
                    {
                        const ValueType alpha = (x - leftKnot) / denom;
                        const ValueType oneMinusAlpha = Constants<ValueType>::one() - alpha;
                        d[j] = oneMinusAlpha * d[j - 1] + alpha * d[j];
                    }
                }
            }

            return d[SplineDegree];
        }

        /**
         * Evaluate the derivative of the B-spline at x.
         *
         * Uses the identity: d/dx B_{i,k}(x) = k * [B_{i,k-1}(x)/(t_{i+k}-t_i) - B_{i+1,k-1}(x)/(t_{i+k+1}-t_{i+1})]
         *
         * Implemented by evaluating two lower-degree splines with adjusted coefficients.
         */
        static ValueType evaluateSplineDerivative(
            const ValueType* coefficients,
            const ValueType* knots,
            const size_t numberOfBasisFunctions,
            const ValueType& x)
        {
            // Derivative coefficients: q_i = k * (c_{i+1} - c_i) / (t_{i+k+1} - t_{i+1})
            // Number of derivative coefficients = numberOfBasisFunctions - 1
            const size_t numDerivCoeffs = numberOfBasisFunctions - 1;
            // Stack array for derivative coefficients (at most GridSize + SplineDegree - 1)
            // We use a reasonable upper bound since numberOfBasisFunctions is runtime
            static const size_t MAX_DERIV_COEFFS = 64;
            ValueType derivCoeffs[MAX_DERIV_COEFFS];

            const ValueType degree = static_cast<ValueType>(SplineDegree);

            for (size_t i = 0; i < numDerivCoeffs; ++i)
            {
                const ValueType denom = knots[i + SplineDegree + 1] - knots[i + 1];
                if (denom > static_cast<ValueType>(0) || denom < static_cast<ValueType>(0))
                {
                    derivCoeffs[i] = degree * (coefficients[i + 1] - coefficients[i]) / denom;
                }
                else
                {
                    derivCoeffs[i] = static_cast<ValueType>(0);
                }
            }

            // Evaluate the degree-(k-1) spline with the derivative coefficients
            // using the interior knots (knots[1] ... knots[NumberOfKnots-2])
            return DeBoorEvaluator<ValueType, SplineDegree - 1>::evaluateSpline(
                derivCoeffs, &knots[1], numDerivCoeffs, x);
        }
    };

    /**
     * Specialization for SplineDegree=1 (piecewise linear).
     *
     * Reduces to simple linear interpolation between grid points.
     * Ideal for fixed-point targets due to minimal rounding error.
     */
    template<typename ValueType>
    struct DeBoorEvaluator<ValueType, 1>
    {
        static size_t findKnotSpan(const ValueType* knots, const size_t numberOfBasisFunctions, const ValueType& x)
        {
            if (x <= knots[1])
            {
                return 1;
            }
            if (x >= knots[numberOfBasisFunctions])
            {
                return numberOfBasisFunctions - 1;
            }

            for (size_t i = 1; i < numberOfBasisFunctions; ++i)
            {
                if (x < knots[i + 1])
                {
                    return i;
                }
            }

            return numberOfBasisFunctions - 1;
        }

        /**
         * For k=1, the spline is piecewise linear.
         * In span [knots[i], knots[i+1]], the value is a linear interpolation
         * between coefficients[i-1] and coefficients[i].
         */
        static ValueType evaluateSpline(
            const ValueType* coefficients,
            const ValueType* knots,
            const size_t numberOfBasisFunctions,
            const ValueType& x)
        {
            const size_t span = findKnotSpan(knots, numberOfBasisFunctions, x);

            const ValueType x0 = knots[span];
            const ValueType x1 = knots[span + 1];
            const ValueType y0 = coefficients[span - 1];
            const ValueType y1 = coefficients[span];

            return linearInterpolation<ValueType>(x, x0, x1, y0, y1);
        }

        /**
         * For k=1, derivative is piecewise constant:
         * In span [knots[i], knots[i+1]], derivative = (c[i] - c[i-1]) / (knots[i+1] - knots[i])
         */
        static ValueType evaluateSplineDerivative(
            const ValueType* coefficients,
            const ValueType* knots,
            const size_t numberOfBasisFunctions,
            const ValueType& x)
        {
            const size_t span = findKnotSpan(knots, numberOfBasisFunctions, x);

            const ValueType x0 = knots[span];
            const ValueType x1 = knots[span + 1];
            const ValueType denom = x1 - x0;

            if (denom > static_cast<ValueType>(0) || denom < static_cast<ValueType>(0))
            {
                return (coefficients[span] - coefficients[span - 1]) / denom;
            }

            return static_cast<ValueType>(0);
        }
    };

    /**
     * Specialization for SplineDegree=0 (piecewise constant).
     * Used internally by the derivative recursion of DeBoorEvaluator<ValueType, 1>.
     */
    template<typename ValueType>
    struct DeBoorEvaluator<ValueType, 0>
    {
        static size_t findKnotSpan(const ValueType* knots, const size_t numberOfBasisFunctions, const ValueType& x)
        {
            if (x <= knots[0])
            {
                return 0;
            }
            if (x >= knots[numberOfBasisFunctions])
            {
                return numberOfBasisFunctions - 1;
            }

            for (size_t i = 0; i < numberOfBasisFunctions; ++i)
            {
                if (x < knots[i + 1])
                {
                    return i;
                }
            }

            return numberOfBasisFunctions - 1;
        }

        static ValueType evaluateSpline(
            const ValueType* coefficients,
            const ValueType* knots,
            const size_t numberOfBasisFunctions,
            const ValueType& x)
        {
            const size_t span = findKnotSpan(knots, numberOfBasisFunctions, x);

            return coefficients[span];
        }

        static ValueType evaluateSplineDerivative(
            const ValueType* coefficients,
            const ValueType* knots,
            const size_t numberOfBasisFunctions,
            const ValueType& x)
        {
            (void)coefficients;
            (void)knots;
            (void)numberOfBasisFunctions;
            (void)x;
            return static_cast<ValueType>(0);
        }
    };
}
