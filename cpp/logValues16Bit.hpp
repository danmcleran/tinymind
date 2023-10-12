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
    #if TINYMIND_USE_LOG_1_15
    struct LogValuesTableQ1_15
    {
        static const uint16_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_LOG_1_15
    #if TINYMIND_USE_LOG_2_14
    struct LogValuesTableQ2_14
    {
        static const uint16_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_LOG_2_14
    #if TINYMIND_USE_LOG_3_13
    struct LogValuesTableQ3_13
    {
        static const uint16_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_LOG_3_13
    #if TINYMIND_USE_LOG_4_12
    struct LogValuesTableQ4_12
    {
        static const uint16_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_LOG_4_12
    #if TINYMIND_USE_LOG_5_11
    struct LogValuesTableQ5_11
    {
        static const uint16_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_LOG_5_11
    #if TINYMIND_USE_LOG_6_10
    struct LogValuesTableQ6_10
    {
        static const uint16_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_LOG_6_10
    #if TINYMIND_USE_LOG_7_9
    struct LogValuesTableQ7_9
    {
        static const uint16_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_LOG_7_9
    #if TINYMIND_USE_LOG_8_8
    struct LogValuesTableQ8_8
    {
        static const uint16_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_LOG_8_8
    #if TINYMIND_USE_LOG_9_7
    struct LogValuesTableQ9_7
    {
        static const uint16_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_LOG_9_7
    #if TINYMIND_USE_LOG_10_6
    struct LogValuesTableQ10_6
    {
        static const uint16_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_LOG_10_6
    #if TINYMIND_USE_LOG_11_5
    struct LogValuesTableQ11_5
    {
        static const uint16_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_LOG_11_5
    #if TINYMIND_USE_LOG_12_4
    struct LogValuesTableQ12_4
    {
        static const uint16_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_LOG_12_4
    #if TINYMIND_USE_LOG_13_3
    struct LogValuesTableQ13_3
    {
        static const uint16_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_LOG_13_3
    #if TINYMIND_USE_LOG_14_2
    struct LogValuesTableQ14_2
    {
        static const uint16_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_LOG_14_2
    #if TINYMIND_USE_LOG_15_1
    struct LogValuesTableQ15_1
    {
        static const uint16_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_LOG_15_1
}
