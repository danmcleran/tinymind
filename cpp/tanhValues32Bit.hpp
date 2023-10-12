/**
* Copyright (c) 2023 Dan McLeran
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

#include "activation.hpp"

namespace tinymind {
    #if TINYMIND_USE_TANH_1_31
    struct TanhValuesTableQ1_31
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_1_31
    #if TINYMIND_USE_TANH_2_30
    struct TanhValuesTableQ2_30
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_2_30
    #if TINYMIND_USE_TANH_3_29
    struct TanhValuesTableQ3_29
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_3_29
    #if TINYMIND_USE_TANH_4_28
    struct TanhValuesTableQ4_28
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_4_28
    #if TINYMIND_USE_TANH_5_27
    struct TanhValuesTableQ5_27
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_5_27
    #if TINYMIND_USE_TANH_6_26
    struct TanhValuesTableQ6_26
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_6_26
    #if TINYMIND_USE_TANH_7_25
    struct TanhValuesTableQ7_25
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_7_25
    #if TINYMIND_USE_TANH_8_24
    struct TanhValuesTableQ8_24
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_8_24
    #if TINYMIND_USE_TANH_9_23
    struct TanhValuesTableQ9_23
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_9_23
    #if TINYMIND_USE_TANH_10_22
    struct TanhValuesTableQ10_22
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_10_22
    #if TINYMIND_USE_TANH_11_21
    struct TanhValuesTableQ11_21
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_11_21
    #if TINYMIND_USE_TANH_12_20
    struct TanhValuesTableQ12_20
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_12_20
    #if TINYMIND_USE_TANH_13_19
    struct TanhValuesTableQ13_19
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_13_19
    #if TINYMIND_USE_TANH_14_18
    struct TanhValuesTableQ14_18
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_14_18
    #if TINYMIND_USE_TANH_15_17
    struct TanhValuesTableQ15_17
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_15_17
    #if TINYMIND_USE_TANH_16_16
    struct TanhValuesTableQ16_16
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_16_16
    #if TINYMIND_USE_TANH_17_15
    struct TanhValuesTableQ17_15
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_17_15
    #if TINYMIND_USE_TANH_18_14
    struct TanhValuesTableQ18_14
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_18_14
    #if TINYMIND_USE_TANH_19_13
    struct TanhValuesTableQ19_13
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_19_13
    #if TINYMIND_USE_TANH_20_12
    struct TanhValuesTableQ20_12
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_20_12
    #if TINYMIND_USE_TANH_21_11
    struct TanhValuesTableQ21_11
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_21_11
    #if TINYMIND_USE_TANH_22_10
    struct TanhValuesTableQ22_10
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_22_10
    #if TINYMIND_USE_TANH_23_9
    struct TanhValuesTableQ23_9
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_23_9
    #if TINYMIND_USE_TANH_24_8
    struct TanhValuesTableQ24_8
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_24_8
    #if TINYMIND_USE_TANH_25_7
    struct TanhValuesTableQ25_7
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_25_7
    #if TINYMIND_USE_TANH_26_6
    struct TanhValuesTableQ26_6
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_26_6
    #if TINYMIND_USE_TANH_27_5
    struct TanhValuesTableQ27_5
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_27_5
    #if TINYMIND_USE_TANH_28_4
    struct TanhValuesTableQ28_4
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_28_4
    #if TINYMIND_USE_TANH_29_3
    struct TanhValuesTableQ29_3
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_29_3
    #if TINYMIND_USE_TANH_30_2
    struct TanhValuesTableQ30_2
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_30_2
    #if TINYMIND_USE_TANH_31_1
    struct TanhValuesTableQ31_1
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_TANH_31_1
}
