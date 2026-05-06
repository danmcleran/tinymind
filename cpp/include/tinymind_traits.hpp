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

/*
 * Minimal type traits for TinyMind's SFINAE dispatches.
 *
 * The standalone composable layers (BatchNorm1D, Dropout, BinaryLayer,
 * TernaryLayer, MaxPool*, BSpline) use enable_if + is_floating_point to
 * pick between a float/double constructor path and a QValue (raw, fractional)
 * constructor path inside their fromInteger() helpers.
 *
 * When TINYMIND_ENABLE_STD=1, these resolve to std::enable_if /
 * std::is_floating_point. When TINYMIND_ENABLE_STD=0, hand-rolled
 * equivalents are provided so headers used by embedded targets do not
 * include <type_traits> or reference namespace std::.
 *
 * Only the subset actually used by TinyMind is provided.
 */

#if TINYMIND_ENABLE_STD
#include <type_traits>
#endif

namespace tinymind {

#if TINYMIND_ENABLE_STD

    template<bool B, typename T = void>
    using enable_if = std::enable_if<B, T>;

    template<typename T>
    using is_floating_point = std::is_floating_point<T>;

#else

    template<bool B, typename T = void>
    struct enable_if {};

    template<typename T>
    struct enable_if<true, T> { typedef T type; };

    template<typename T>
    struct is_floating_point { static const bool value = false; };

    template<> struct is_floating_point<float>       { static const bool value = true; };
    template<> struct is_floating_point<double>      { static const bool value = true; };
    template<> struct is_floating_point<long double> { static const bool value = true; };

    template<typename T> struct is_floating_point<const T>          : is_floating_point<T> {};
    template<typename T> struct is_floating_point<volatile T>       : is_floating_point<T> {};
    template<typename T> struct is_floating_point<const volatile T> : is_floating_point<T> {};

#endif

}
