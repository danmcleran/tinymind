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
 * Phase 14 threading helper.
 *
 * Provides TINYMIND_PARALLEL_FOR_OUTER, which expands to an OpenMP
 * pragma when TINYMIND_ENABLE_OPENMP=1 and to nothing otherwise. The
 * caller still writes a plain `for` immediately after the macro; the
 * pragma simply annotates that loop.
 *
 * Orthogonal to every SIMD gate. The caller is responsible for passing
 * `-fopenmp` (or equivalent) to the compiler when the gate is on; this
 * header does not include <omp.h>.
 *
 * Usage:
 *   TINYMIND_PARALLEL_FOR_OUTER
 *   for (std::size_t f = 0; f < NumFilters; ++f) { ... }
 *
 * The pragma applies to the output-channel / output-filter dimension of
 * conv layers, which is the only loop whose iterations write to
 * non-overlapping output regions. Inner spatial / kernel / channel
 * loops are not parallelized to keep the cost model predictable.
 */

#include "tinymind_platform.hpp"

#if TINYMIND_ENABLE_OPENMP
#  define TINYMIND_PRAGMA(x) _Pragma(#x)
#  define TINYMIND_PARALLEL_FOR_OUTER TINYMIND_PRAGMA(omp parallel for)
#else
#  define TINYMIND_PARALLEL_FOR_OUTER
#endif
