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
 * Phase 14 SIMD backend: SVE2 superset.
 *
 * Gate: TINYMIND_ENABLE_SIMD_SVE2. Requires SIMD_SVE (and transitively
 * SIMD_NEON). SVE2 adds NEON-equivalent integer ops to SVE, completing
 * the integer ISA story for vector-length-agnostic code on Neoverse V2 /
 * N2 and other Armv9-A application cores.
 *
 * The int8 dot product primitive itself only needs SDOT, which already
 * ships with SVE(1); the dispatch defers to the SVE backend so SVE2
 * targets do not pay for a second implementation when nothing in SVE2
 * is exploitable here. Future SVE2-only primitives (e.g. histogram
 * acceleration in qsoftmax) will land in this header.
 */

#include "../tinymind_platform.hpp"

#if TINYMIND_ENABLE_SIMD_SVE2

static_assert(TINYMIND_ENABLE_SIMD_SVE,
              "TINYMIND_ENABLE_SIMD_SVE2 requires TINYMIND_ENABLE_SIMD_SVE. "
              "SVE2 is a strict superset of SVE.");

#endif // TINYMIND_ENABLE_SIMD_SVE2
