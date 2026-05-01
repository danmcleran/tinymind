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

/*
 * Platform feature gates for TinyMind.
 *
 * All TINYMIND_ENABLE_* macros default to 0 so that a freestanding embedded
 * build (no FPU, no full C++ stdlib) compiles out of the box. Hosted users
 * opt in via -DTINYMIND_ENABLE_FLOAT=1, etc.
 *
 * TINYMIND_ENABLE_OSTREAMS  - <ostream> support in qformat.hpp and other
 *                             debug-printing paths.
 * TINYMIND_ENABLE_FLOAT     - float/double-typed code paths: Adam/RMSprop
 *                             float optimizers, Xavier initializer, and the
 *                             float/double specializations of ValueParser /
 *                             ValueConverter in nnproperties.hpp. These
 *                             paths call <cmath> functions (std::sqrt,
 *                             std::pow) that require an FPU or softfp.
 * TINYMIND_ENABLE_HOSTED_IO - File I/O serialization in nnproperties.hpp
 *                             (NetworkPropertiesFileManager and friends),
 *                             which pulls in <fstream> and <vector>.
 */

#ifndef TINYMIND_ENABLE_OSTREAMS
#define TINYMIND_ENABLE_OSTREAMS 0
#endif

#ifndef TINYMIND_ENABLE_FLOAT
#define TINYMIND_ENABLE_FLOAT 0
#endif

#ifndef TINYMIND_ENABLE_HOSTED_IO
#define TINYMIND_ENABLE_HOSTED_IO 0
#endif
