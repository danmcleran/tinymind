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
 * build (no FPU, no full C++ stdlib, no C runtime rand()) compiles out of
 * the box. Hosted users opt in via -DTINYMIND_ENABLE_FLOAT=1, etc.
 *
 * The five gates are orthogonal — float-as-ValueType, namespace std::,
 * file I/O, ostream, and rand() are independently available on different
 * embedded toolchains.
 *
 * TINYMIND_ENABLE_FLOAT       - float/double as ValueType: float/double
 *                               specializations of ValueParser /
 *                               ValueConverter, plus float optimizers and
 *                               Xavier initialization. Requires float
 *                               arithmetic (FPU or softfp). Does NOT by
 *                               itself pull in <cmath>; only sites that
 *                               actually need transcendental functions
 *                               (sqrt, pow) do, gated additionally by
 *                               TINYMIND_ENABLE_STD.
 *
 * TINYMIND_ENABLE_STD         - namespace std:: and standard headers
 *                               (<cmath>, <type_traits>) are available.
 *                               Off by default for embedded toolchains
 *                               that ship a stripped-down or alternate
 *                               C++ stdlib. When off, TinyMind uses its
 *                               own minimal trait equivalents in
 *                               tinymind_traits.hpp and avoids any
 *                               <cmath> dependency.
 *
 * TINYMIND_ENABLE_HOSTED_IO   - File I/O serialization in nnproperties.hpp
 *                               (NetworkPropertiesFileManager and friends),
 *                               which pulls in <fstream> and <vector>, plus
 *                               <cstdlib> / <cstdio> for parsing helpers.
 *
 * TINYMIND_ENABLE_OSTREAMS    - <ostream> support in qformat.hpp and other
 *                               debug-printing paths.
 *
 * TINYMIND_ENABLE_HOSTED_RAND - <cstdlib> rand() / RAND_MAX availability,
 *                               used by Dropout, ScheduledSampling, and
 *                               Xavier initializers. On freestanding
 *                               targets without a C runtime, leave off
 *                               and supply your own RNG at the call site.
 *
 * TINYMIND_ENABLE_QUANTIZATION - Post-training int8 quantization path
 *                                (qaffine.hpp + parallel Q* layer family).
 *                                Distinct from QValue Q-format: introduces
 *                                runtime per-tensor scale/zero-point and
 *                                int32 accumulators with integer
 *                                requantization between layers. Off by
 *                                default; existing fixed-point and float
 *                                pipelines are unaffected.
 *
 * TINYMIND_ENABLE_FP16        - Half-precision storage tier (fp16 + bf16).
 *                                Provides software IEEE-754 binary16 and
 *                                bfloat16 storage types in
 *                                tinymind_fp16.hpp, with float promotion
 *                                helpers used by qbridge.hpp. Storage only
 *                                in this phase; no SIMD specializations.
 *                                Requires TINYMIND_ENABLE_FLOAT for the
 *                                conversion paths.
 *
 * TINYMIND_ENABLE_INT16_ACCUM - Wider-precision carry-state for the
 *                                quantized recurrent layers (QLSTM /
 *                                QGRU). When on, QLSTM cells and
 *                                long-sequence accumulator paths can be
 *                                instantiated with int16 cell-state
 *                                storage to retain dynamic range across
 *                                long unroll horizons. Pure storage gate;
 *                                no SIMD implications. Off by default to
 *                                keep deployable footprints tight.
 */

#ifndef TINYMIND_ENABLE_FLOAT
#define TINYMIND_ENABLE_FLOAT 0
#endif

#ifndef TINYMIND_ENABLE_STD
#define TINYMIND_ENABLE_STD 0
#endif

#ifndef TINYMIND_ENABLE_HOSTED_IO
#define TINYMIND_ENABLE_HOSTED_IO 0
#endif

#ifndef TINYMIND_ENABLE_OSTREAMS
#define TINYMIND_ENABLE_OSTREAMS 0
#endif

#ifndef TINYMIND_ENABLE_HOSTED_RAND
#define TINYMIND_ENABLE_HOSTED_RAND 0
#endif

#ifndef TINYMIND_ENABLE_QUANTIZATION
#define TINYMIND_ENABLE_QUANTIZATION 0
#endif

#ifndef TINYMIND_ENABLE_FP16
#define TINYMIND_ENABLE_FP16 0
#endif

#ifndef TINYMIND_ENABLE_INT16_ACCUM
#define TINYMIND_ENABLE_INT16_ACCUM 0
#endif
