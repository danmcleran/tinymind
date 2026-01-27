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

// MIN_X_TABLE_VALUE, MAX_X_TABLE_VALUE, ACTIVATION_DELTA_SHIFT & NUMBER_OF_ACTIVATION_TABLE_VALUES
// are configurable definitions for the activation function lookup table. They can be defined via
// compiler flags before including this header file, otherwise default values are used as shown
// below. Makefile in apps/activation shows how these values are currently defined via compiler
// flags with -D option.

#ifndef MIN_X_TABLE_VALUE
#define MIN_X_TABLE_VALUE -6  // Default min x value for activation table if not defined
#endif

#ifndef MAX_X_TABLE_VALUE
#define MAX_X_TABLE_VALUE 6   // Default max x value for activation table if not defined
#endif

#ifndef ACTIVATION_DELTA_SHIFT
#define ACTIVATION_DELTA_SHIFT 3  // Default delta shift for activation table if not defined
#endif

#ifndef NUMBER_OF_ACTIVATION_TABLE_VALUES
#define NUMBER_OF_ACTIVATION_TABLE_VALUES 96  // Default number of activation table values if not defined
#endif

#define NUMBER_OF_ACTIVATION_TABLE_CHK_VALUES  ((MAX_X_TABLE_VALUE - MIN_X_TABLE_VALUE) * (1 << (ACTIVATION_DELTA_SHIFT)))

static_assert(MIN_X_TABLE_VALUE >= -16, "MIN_X_TABLE_VALUE must be >= -16"); // Smallest value for min x table value
static_assert(MIN_X_TABLE_VALUE <= -6, "MIN_X_TABLE_VALUE must be <= -6");  // Largest value for min x table value

static_assert(MAX_X_TABLE_VALUE >= 6, "MAX_X_TABLE_VALUE must be >= 6");  // Smallest value for max x table value
static_assert(MAX_X_TABLE_VALUE <= 16, "MAX_X_TABLE_VALUE must be <= 16");  // Largest value for max x table value

static_assert(ACTIVATION_DELTA_SHIFT >= 3, "ACTIVATION_DELTA_SHIFT must be >= 3"); // Smallest value for delta shift
static_assert(ACTIVATION_DELTA_SHIFT <= 4, "ACTIVATION_DELTA_SHIFT must be <= 4"); // Largest value for delta shift

// Sanity check for the correct number of activation table values based on this caculation: (MAX - MIN) * 2^ACTIVATION_DELTA_SHIFT
static_assert((NUMBER_OF_ACTIVATION_TABLE_VALUES == NUMBER_OF_ACTIVATION_TABLE_CHK_VALUES), "Invalid # of activation table values");
