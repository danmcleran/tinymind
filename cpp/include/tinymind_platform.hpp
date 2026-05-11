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
 *
 * Phase 14 SIMD capability gates. Each names an ISA extension, never a
 * CPU model. The scalar fallback is the default body of every layer, so
 * leaving all gates off produces the same bytes the pre-Phase-14 build
 * emitted. Gates honor Arm's architectural prerequisite chain (each
 * dependent gate's header opens with a static_assert that the
 * prerequisite is also on); misconfiguration fails at compile time
 * rather than producing silent miscompiles.
 *
 * TINYMIND_ENABLE_SIMD_NEON          - Armv7 / Armv8-A Advanced SIMD
 *                                       baseline (VMULL, VPADDL). Implicit
 *                                       prerequisite for every other Arm
 *                                       application-class gate.
 * TINYMIND_ENABLE_SIMD_NEON_DOTPROD  - Armv8.2-A FEAT_DotProd (SDOT/UDOT).
 *                                       Requires SIMD_NEON. Headline int8
 *                                       MAC accelerator on Arm.
 * TINYMIND_ENABLE_SIMD_NEON_FP16     - Armv8.2-A FEAT_FP16 vector forms.
 *                                       Requires SIMD_NEON. Pairs with the
 *                                       Phase 9 fp16_t storage tier.
 * TINYMIND_ENABLE_SIMD_SVE           - Scalable Vector Extension,
 *                                       vector-length-agnostic loops.
 *                                       Requires SIMD_NEON (per GCC
 *                                       AArch64 docs: "+sve also enables
 *                                       Advanced SIMD and floating-point
 *                                       instructions").
 * TINYMIND_ENABLE_SIMD_SVE2          - SVE2 superset. Requires SIMD_SVE.
 * TINYMIND_ENABLE_SIMD_HELIUM_MVE_I  - Armv8.1-M MVE-I (integer Helium).
 *                                       M-profile only; mutually
 *                                       exclusive with SIMD_NEON / SVE.
 * TINYMIND_ENABLE_SIMD_HELIUM_MVE_F  - Armv8.1-M MVE-F (float Helium).
 *                                       Independent of MVE-I per Arm
 *                                       Helium docs (a core may
 *                                       implement either alone).
 * TINYMIND_ENABLE_SIMD_AVX2          - x86 AVX2 + SSSE3 baseline
 *                                       (PMADDUBSW path).
 * TINYMIND_ENABLE_SIMD_AVX_VNNI      - 256-bit AVX-VNNI (VPDPBUSD on
 *                                       Alder Lake+). Requires
 *                                       SIMD_AVX2. Note: AVX-VNNI is
 *                                       independent of AVX-512 VNNI; a
 *                                       CPU may ship one without the
 *                                       other.
 * TINYMIND_ENABLE_SIMD_AVX512F       - AVX-512 foundation.
 * TINYMIND_ENABLE_SIMD_AVX512_VNNI   - AVX-512-VNNI. Requires
 *                                       SIMD_AVX512F.
 *
 * TINYMIND_ENABLE_OPENMP             - Optional `#pragma omp parallel
 *                                       for` over the output-channel
 *                                       dim of conv layers. Orthogonal
 *                                       to every SIMD gate. Caller must
 *                                       pass `-fopenmp` to the compiler.
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

#ifndef TINYMIND_ENABLE_SIMD_NEON
#define TINYMIND_ENABLE_SIMD_NEON 0
#endif

#ifndef TINYMIND_ENABLE_SIMD_NEON_DOTPROD
#define TINYMIND_ENABLE_SIMD_NEON_DOTPROD 0
#endif

#ifndef TINYMIND_ENABLE_SIMD_NEON_FP16
#define TINYMIND_ENABLE_SIMD_NEON_FP16 0
#endif

#ifndef TINYMIND_ENABLE_SIMD_SVE
#define TINYMIND_ENABLE_SIMD_SVE 0
#endif

#ifndef TINYMIND_ENABLE_SIMD_SVE2
#define TINYMIND_ENABLE_SIMD_SVE2 0
#endif

#ifndef TINYMIND_ENABLE_SIMD_HELIUM_MVE_I
#define TINYMIND_ENABLE_SIMD_HELIUM_MVE_I 0
#endif

#ifndef TINYMIND_ENABLE_SIMD_HELIUM_MVE_F
#define TINYMIND_ENABLE_SIMD_HELIUM_MVE_F 0
#endif

#ifndef TINYMIND_ENABLE_SIMD_AVX2
#define TINYMIND_ENABLE_SIMD_AVX2 0
#endif

#ifndef TINYMIND_ENABLE_SIMD_AVX_VNNI
#define TINYMIND_ENABLE_SIMD_AVX_VNNI 0
#endif

#ifndef TINYMIND_ENABLE_SIMD_AVX512F
#define TINYMIND_ENABLE_SIMD_AVX512F 0
#endif

#ifndef TINYMIND_ENABLE_SIMD_AVX512_VNNI
#define TINYMIND_ENABLE_SIMD_AVX512_VNNI 0
#endif

#ifndef TINYMIND_ENABLE_OPENMP
#define TINYMIND_ENABLE_OPENMP 0
#endif
