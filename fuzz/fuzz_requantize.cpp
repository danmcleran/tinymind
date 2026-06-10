/*
 * libFuzzer harness for multiplyByQuantizedMultiplier -- the requantization
 * primitive shared by every int8 layer (dense, conv, attention, the output
 * stage of layernorm/softmax/fft).
 *
 * Rationale: this single function is the most-reused arithmetic in the int8
 * tier. It left-shifts, calls saturatingRoundingDoublingHighMul, then
 * roundingDivideByPOT. Each step has signed-overflow corners that only fire on
 * specific (value, multiplier, shift) triples: INT32_MIN inputs, the
 * (-2^31)*(-2^31) doubling-high-mul rail, and large positive/negative shifts.
 * A fixed-input unit test cannot enumerate the triple space; the fuzzer can.
 *
 * Inputs are drawn across the full int32 domain, with shift constrained to the
 * realistic [-31, 31] window the importers emit, plus the calibrated
 * "normalized multiplier" band [2^30, 2^31) that real quantization produces.
 *
 * Build/run: see fuzz/Makefile (`make fuzz` / `make fuzz-ci`).
 *
 * Copyright (c) 2026 Dan McLeran. MIT (see LICENSE.txt).
 */
#include <cstddef>
#include <cstdint>

#include "qaffine.hpp"

using tinymind::multiplyByQuantizedMultiplier;

namespace {

// Linear byte cursor. Reads past the end return 0.
struct ByteReader
{
    const uint8_t* p;
    size_t n;
    size_t i;

    explicit ByteReader(const uint8_t* data, size_t size) : p(data), n(size), i(0) {}

    uint8_t u8() { return (i < n) ? p[i++] : static_cast<uint8_t>(0); }

    int32_t i32()
    {
        uint32_t v = 0;
        for (int b = 0; b < 4; ++b) { v = (v << 8) | u8(); }
        return static_cast<int32_t>(v);
    }
};

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    ByteReader br(data, size);

    const int32_t value = br.i32();

    // Two multiplier regimes selected by the first bit: the full int32 domain
    // (adversarial, incl. INT32_MIN), and the calibrated normalized band
    // [2^30, 2^31) that real per-tensor quantization emits.
    int32_t multiplier;
    if (br.u8() & 0x01)
    {
        multiplier = br.i32();
    }
    else
    {
        const uint32_t span = 0x40000000u; // 2^30 values up to just under 2^31
        multiplier = static_cast<int32_t>(0x40000000u + (static_cast<uint32_t>(br.i32()) % span));
    }

    // Importer-reachable shift band. shift = -frexp-exponent of the scale
    // ratio (in_scale * w_scale / out_scale); real int8 calibration keeps that
    // ratio within ~[1e-3, 1e2], so shift lands in roughly [-8, 31]. left_shift
    // (negative shift) thus stays small -- the regime the primitive supports.
    int32_t shift = (static_cast<int32_t>(br.u8()) % 40) - 8; // [-8, 31]

    const int32_t out = multiplyByQuantizedMultiplier(value, multiplier, shift);

    volatile int32_t sink = out;
    (void)sink;
    return 0;
}
