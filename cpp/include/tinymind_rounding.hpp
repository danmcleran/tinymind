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

#include "tinymind_platform.hpp"

/**
 * Rounding of the products an SGD step is built from, whatever a fixed-point
 * format's own rounding policy.
 *
 * - learningRate x gradient is rounded to nearest (ties away from zero).
 *   The default TruncatePolicy floors, so a step smaller than one LSB was 0
 *   when positive and a full -1 LSB when negative: every weight with a small
 *   gradient drifted down by about half an LSB per step. In Q8.8 that swamps
 *   the gradient (the xor example built in Q8.8 trained on 25 of 100 seeds
 *   with it, 100 of 100 without). Rounding toward zero removes the drift too,
 *   but drops every step below one LSB, and recurrent Q16.16 training with a
 *   small learning rate stalls on that dead zone; to nearest the dead zone is
 *   half an LSB.
 *
 * - momentum x (last delta) and acceleration x (the delta before) are rounded
 *   toward zero, so a delta that no gradient refreshes decays to exactly 0.
 *   Under RoundUpPolicy 0.5 x (+1 LSB) rounds back up to +1 LSB, and under
 *   TruncatePolicy 0.5 x (-1 LSB) back down to -1 LSB, so a one-LSB delta
 *   reproduced itself forever and moved its weight by an LSB on every step;
 *   to nearest has the same fixed point.
 *
 * These are the fallbacks for floating-point (and any other) types: they
 * multiply as usual. The QValue overloads live in qformat.hpp and are found
 * by argument-dependent lookup, so layers that only need these declarations
 * do not have to include qformat.hpp.
 */
namespace tinymind {
    template<typename ValueType>
    ValueType multiplyRoundNearest(const ValueType& left, const ValueType& right)
    {
        return left * right;
    }

    template<typename ValueType>
    ValueType multiplyRoundTowardZero(const ValueType& left, const ValueType& right)
    {
        return left * right;
    }
}
