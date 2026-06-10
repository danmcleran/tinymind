/*
 * libFuzzer differential harness: AVX2 int8 dot-product backend vs the
 * scalar reference.
 *
 * Rationale: tinymind::simd::int8DotWithZeroPoint is the single seam through
 * which every quantized layer (QDense, QConv2D, depthwise/pointwise) reaches
 * the active SIMD backend -- all layer code above it is identical between
 * scalar and SIMD builds. The dispatch contract says every backend is
 * bit-exact with int8DotWithZeroPointScalar; the AVX2 implementation also
 * rearranges the zero-point algebraically (sum(w*x) - zp*sum(w) instead of
 * sum(w*(x-zp))) and handles the n%16 tail in scalar code, so the corners are
 * exactly the ones mutation finds well: short buffers, ragged tails,
 * full-rail +/-127/-128 operands, extreme zero points. A mismatch traps, a
 * memory bug in the intrinsics path aborts under ASan -- both surface as
 * crashes.
 *
 * This harness must be compiled with -mavx2 and TINYMIND_ENABLE_SIMD_AVX2=1
 * (see fuzz/Makefile); both the scalar reference and the avx2:: namespace are
 * then visible in one binary, which is what makes the differential possible.
 * Lengths are capped so the raw sum(w*x) intermediate cannot overflow int32
 * -- the rearrangement is only contract-equal in that regime, which real
 * layer dimensions stay far inside.
 *
 * Build/run: see fuzz/Makefile (`make fuzz` / `make fuzz-ci`).
 *
 * Copyright (c) 2026 Dan McLeran. MIT (see LICENSE.txt).
 */
#include <cstddef>
#include <cstdint>

#include "include/simd/simd_dispatch.hpp"

#if !TINYMIND_ENABLE_SIMD_AVX2
#error "fuzz_simd_avx2_diff requires -DTINYMIND_ENABLE_SIMD_AVX2=1 -mavx2"
#endif

using tinymind::simd::int8DotWithZeroPointScalar;

namespace {

// Linear byte cursor. Reads past the end return 0.
struct ByteReader
{
    const uint8_t* p;
    size_t n;
    size_t i;

    explicit ByteReader(const uint8_t* data, size_t size) : p(data), n(size), i(0) {}

    uint8_t u8() { return (i < n) ? p[i++] : static_cast<uint8_t>(0); }
    int8_t  i8() { return static_cast<int8_t>(u8()); }
};

// Large enough to cover several 16-lane AVX2 iterations plus every tail
// residue; small enough that |sum(w*x)| <= 80*127*128 stays far from the
// int32 rails.
constexpr std::size_t kMaxLen = 80;
// Slack so a fuzz-chosen offset can misalign the buffers against the 16-byte
// vector loads (the backend uses loadu, but ASan checks the touched bytes).
constexpr std::size_t kAlignSlack = 31;

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    ByteReader br(data, size);

    // Length spans 0 (empty: pure-tail path), sub-vector, and multi-vector
    // with every tail residue mod 16.
    const std::size_t n = static_cast<std::size_t>(br.u8()) % (kMaxLen + 1);
    // Independent misalignments for x and w.
    const std::size_t x_off = static_cast<std::size_t>(br.u8()) % (kAlignSlack + 1);
    const std::size_t w_off = static_cast<std::size_t>(br.u8()) % (kAlignSlack + 1);
    const int8_t zp = br.i8();

    int8_t x_buf[kMaxLen + kAlignSlack];
    int8_t w_buf[kMaxLen + kAlignSlack];
    int8_t* x = x_buf + x_off;
    int8_t* w = w_buf + w_off;
    for (std::size_t k = 0; k < n; ++k) { x[k] = br.i8(); }
    for (std::size_t k = 0; k < n; ++k) { w[k] = br.i8(); }

    const int32_t simd_out   = tinymind::simd::avx2::int8DotWithZeroPoint(x, w, n, zp);
    const int32_t scalar_out = int8DotWithZeroPointScalar(x, w, n, zp);

    if (simd_out != scalar_out)
    {
        __builtin_trap();
    }
    return 0;
}
