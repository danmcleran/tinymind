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

#include "activation.hpp"

namespace tinymind {
    #if TINYMIND_USE_COS_1_7
    struct CosValuesTableQ1_7
    {
        static const uint8_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_1_7
    #if TINYMIND_USE_COS_2_6
    struct CosValuesTableQ2_6
    {
        static const uint8_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_2_6
    #if TINYMIND_USE_COS_3_5
    struct CosValuesTableQ3_5
    {
        static const uint8_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_3_5
    #if TINYMIND_USE_COS_4_4
    struct CosValuesTableQ4_4
    {
        static const uint8_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_4_4
    #if TINYMIND_USE_COS_5_3
    struct CosValuesTableQ5_3
    {
        static const uint8_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_5_3
    #if TINYMIND_USE_COS_6_2
    struct CosValuesTableQ6_2
    {
        static const uint8_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_6_2
    #if TINYMIND_USE_COS_7_1
    struct CosValuesTableQ7_1
    {
        static const uint8_t values[NUMBER_OF_ACTIVATION_TABLE_VALUES];
    };
    #endif // TINYMIND_USE_COS_7_1
}
