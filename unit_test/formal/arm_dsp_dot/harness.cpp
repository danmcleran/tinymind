/*
 * CBMC equivalence proof: the SMLAD backend of int8DotWithZeroPoint returns
 * the same int32 as the scalar reference, for every input of length
 * 0..TM_MAX_N, every zero point, and every buffer alignment.
 */
#include "include/tinymind_platform.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>

// CBMC's C++ front end has no model for __builtin_memcpy: the call is
// skipped and load4() returns an unconstrained word, which yields spurious
// counterexamples. Its library does model memcpy, so route the builtin there.
// The Makefile also fails the run on any "no body for function" warning.
#ifndef TM_NO_MEMCPY_SHIM
#define __builtin_memcpy std::memcpy
#endif

// Select the Clang source path of simd_arm_dsp.hpp: it spells the rotate in
// C, which CBMC can reason about, where the GCC path uses inline asm.
#define __clang__ 1
#include "include/simd/simd_arm_dsp.hpp"
#undef __clang__

#include "include/simd/simd_dispatch.hpp"

#ifndef TM_XOFF
#define TM_XOFF 0
#endif
#ifndef TM_WOFF
#define TM_WOFF 0
#endif

int8_t nondet_int8();

int main()
{
    // Length and buffer offsets are compile-time constants, one CBMC run per
    // (n, xoff, woff) triple: the solver then only has to reason about the
    // data, which is fully symbolic. Spare bytes let x and w start at any
    // offset mod 4.
    const std::size_t n = TM_N;
    int8_t xbuf[TM_N + 3];
    int8_t wbuf[TM_N + 3];
    for (std::size_t i = 0; i < TM_N + 3; ++i)
    {
        xbuf[i] = nondet_int8();
        wbuf[i] = nondet_int8();
    }
    const int8_t zp = nondet_int8();

    const int8_t* x = xbuf + TM_XOFF;
    const int8_t* w = wbuf + TM_WOFF;

    const int32_t simd = tinymind::simd::arm_dsp::int8DotWithZeroPoint(x, w, n, zp);
    const int32_t ref = tinymind::simd::int8DotWithZeroPointScalar(x, w, n, zp);

    __CPROVER_assert(simd == ref, "SMLAD backend equals scalar reference");
    return 0;
}
