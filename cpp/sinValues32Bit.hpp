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
    #if TINYMIND_USE_SIN_1_31
    struct SinValuesTableQ1_31
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_1_31
    #if TINYMIND_USE_SIN_2_30
    struct SinValuesTableQ2_30
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_2_30
    #if TINYMIND_USE_SIN_3_29
    struct SinValuesTableQ3_29
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_3_29
    #if TINYMIND_USE_SIN_4_28
    struct SinValuesTableQ4_28
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_4_28
    #if TINYMIND_USE_SIN_5_27
    struct SinValuesTableQ5_27
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_5_27
    #if TINYMIND_USE_SIN_6_26
    struct SinValuesTableQ6_26
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_6_26
    #if TINYMIND_USE_SIN_7_25
    struct SinValuesTableQ7_25
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_7_25
    #if TINYMIND_USE_SIN_8_24
    struct SinValuesTableQ8_24
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_8_24
    #if TINYMIND_USE_SIN_9_23
    struct SinValuesTableQ9_23
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_9_23
    #if TINYMIND_USE_SIN_10_22
    struct SinValuesTableQ10_22
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_10_22
    #if TINYMIND_USE_SIN_11_21
    struct SinValuesTableQ11_21
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_11_21
    #if TINYMIND_USE_SIN_12_20
    struct SinValuesTableQ12_20
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_12_20
    #if TINYMIND_USE_SIN_13_19
    struct SinValuesTableQ13_19
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_13_19
    #if TINYMIND_USE_SIN_14_18
    struct SinValuesTableQ14_18
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_14_18
    #if TINYMIND_USE_SIN_15_17
    struct SinValuesTableQ15_17
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_15_17
    #if TINYMIND_USE_SIN_16_16
    struct SinValuesTableQ16_16
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_16_16
    #if TINYMIND_USE_SIN_17_15
    struct SinValuesTableQ17_15
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_17_15
    #if TINYMIND_USE_SIN_18_14
    struct SinValuesTableQ18_14
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_18_14
    #if TINYMIND_USE_SIN_19_13
    struct SinValuesTableQ19_13
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_19_13
    #if TINYMIND_USE_SIN_20_12
    struct SinValuesTableQ20_12
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_20_12
    #if TINYMIND_USE_SIN_21_11
    struct SinValuesTableQ21_11
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_21_11
    #if TINYMIND_USE_SIN_22_10
    struct SinValuesTableQ22_10
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_22_10
    #if TINYMIND_USE_SIN_23_9
    struct SinValuesTableQ23_9
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_23_9
    #if TINYMIND_USE_SIN_24_8
    struct SinValuesTableQ24_8
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_24_8
    #if TINYMIND_USE_SIN_25_7
    struct SinValuesTableQ25_7
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_25_7
    #if TINYMIND_USE_SIN_26_6
    struct SinValuesTableQ26_6
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_26_6
    #if TINYMIND_USE_SIN_27_5
    struct SinValuesTableQ27_5
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_27_5
    #if TINYMIND_USE_SIN_28_4
    struct SinValuesTableQ28_4
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_28_4
    #if TINYMIND_USE_SIN_29_3
    struct SinValuesTableQ29_3
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_29_3
    #if TINYMIND_USE_SIN_30_2
    struct SinValuesTableQ30_2
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_30_2
    #if TINYMIND_USE_SIN_31_1
    struct SinValuesTableQ31_1
    {
        static const uint32_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_SIN_31_1
}
