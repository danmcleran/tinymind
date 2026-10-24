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
 * importer-reachable [-8, 31] band, plus the calibrated "normalized
 * multiplier" band [2^30, 2^31) that real quantization produces.
 *
 * Properties checked on every input (a failure traps):
 *   1. multiplyByQuantizedMultiplier equals an exact reference written from
 *      the rounding definitions (saturating left shift, round-half-up
 *      doubling high multiply, round-half-away-from-zero divide), computed
 *      in 128-bit arithmetic with divisions rather than the kernel's shifts.
 *   2. Requantizer<int32_t, int8_t / uint8_t>::apply -- whose common case
 *      takes the one-rounding roundingHighMulDivideByPOT fast path --
 *      equals that reference plus the zero point, clamped to [qmin, qmax].
 *   3. Whenever FastRequantizer::init accepts a requantizer, its apply is
 *      bit-identical to Requantizer::apply; and init must accept every
 *      requantizer its documented contract covers.
 *
 * Build/run: see fuzz/Makefile (`make fuzz` / `make fuzz-ci`).
 *
 * Copyright (c) 2026 Dan McLeran. MIT (see LICENSE.txt).
 */
#include <cstddef>
#include <cstdint>

#include "qaffine.hpp"
#include "fuzz_check.hpp"

using tinymind::multiplyByQuantizedMultiplier;
using tinymind::Requantizer;
using tinymind::detail::FastRequantizer;

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

// __extension__ keeps -Wpedantic quiet about the non-ISO type.
__extension__ typedef __int128 Wide;

// floor(a / b) for b > 0, by division, not by an arithmetic shift.
Wide floorDiv(Wide a, Wide b)
{
    Wide q = a / b;
    if ((a % b != 0) && (a < 0))
    {
        --q;
    }
    return q;
}

// The requantization primitive written from its definition, independent of
// the kernel's shift-and-mask formulation.
int32_t referenceMultiply(int32_t x, int32_t multiplier, int32_t shift)
{
    const int32_t left_shift  = (shift < 0) ? -shift : 0;
    const int32_t right_shift = (shift > 0) ?  shift : 0;

    // Left shift, saturated to int32.
    Wide shifted = static_cast<Wide>(x) * (static_cast<Wide>(1) << left_shift);
    if (shifted > INT32_MAX) { shifted = INT32_MAX; }
    if (shifted < INT32_MIN) { shifted = INT32_MIN; }

    // Doubling high multiply: round(s * m / 2^31), ties toward +inf, with the
    // single (-2^31) * (-2^31) corner saturating to INT32_MAX.
    Wide high;
    if ((shifted == INT32_MIN) && (multiplier == INT32_MIN))
    {
        high = INT32_MAX;
    }
    else
    {
        const Wide two31 = static_cast<Wide>(1) << 31;
        high = floorDiv(shifted * static_cast<Wide>(multiplier) + (two31 >> 1), two31);
    }

    // Divide by 2^right_shift, ties away from zero.
    if (right_shift == 0)
    {
        return static_cast<int32_t>(high);
    }
    const Wide divisor = static_cast<Wide>(1) << right_shift;
    const Wide magnitude = (high < 0) ? -high : high;
    const Wide rounded = (magnitude + (divisor >> 1)) / divisor;
    return static_cast<int32_t>((high < 0) ? -rounded : rounded);
}

template<typename Dst>
void checkRequantizer(int32_t acc, int32_t multiplier, int32_t shift, int32_t expected_scaled,
                      ByteReader& br)
{
    const int32_t storage_lo = tinymind::detail::StorageRange<Dst>::min;
    const int32_t storage_hi = tinymind::detail::StorageRange<Dst>::max;

    Requantizer<int32_t, Dst> rq;
    rq.multiplier = multiplier;
    rq.shift      = shift;
    rq.zero_point = static_cast<Dst>(storage_lo + br.u8());
    if (br.u8() & 0x01)
    {
        // A fused-activation clamp narrower than the storage range.
        int32_t a = storage_lo + br.u8();
        int32_t b = storage_lo + br.u8();
        if (a > b) { const int32_t t = a; a = b; b = t; }
        rq.qmin = static_cast<Dst>(a);
        rq.qmax = static_cast<Dst>(b);
    }
    else
    {
        rq.qmin = static_cast<Dst>(storage_lo);
        rq.qmax = static_cast<Dst>(storage_hi);
    }

    const int32_t got = static_cast<int32_t>(rq.apply(acc));

    // The reference result plus the zero point can leave int32 only far
    // outside the storage range, where both clamp to the same rail.
    int64_t want = static_cast<int64_t>(expected_scaled) + static_cast<int64_t>(rq.zero_point);
    if (want < static_cast<int64_t>(rq.qmin)) { want = rq.qmin; }
    if (want > static_cast<int64_t>(rq.qmax)) { want = rq.qmax; }
    TINYMIND_FUZZ_CHECK(got == static_cast<int32_t>(want));
    tinymind_fuzz::emitValue(got);

    FastRequantizer<Dst> fast;
    const bool accepted = fast.init(rq);
    const bool full_range = (static_cast<int32_t>(rq.qmin) == storage_lo) &&
                            (static_cast<int32_t>(rq.qmax) == storage_hi);
    const int64_t zp_term = static_cast<int64_t>(rq.zero_point) *
                            ((shift >= 1 && shift <= 31) ? (static_cast<int64_t>(1) << (shift - 1)) : 0);
    const bool in_contract = (multiplier >= 0) && (shift >= 2) && (shift <= 31) && full_range &&
                             (zp_term <= (static_cast<int64_t>(1) << 29)) &&
                             (zp_term >= -(static_cast<int64_t>(1) << 29));
    TINYMIND_FUZZ_CHECK(accepted == in_contract);
    if (accepted)
    {
        TINYMIND_FUZZ_CHECK(static_cast<int32_t>(fast.apply(acc)) == got);
    }
}

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
    const int32_t want = referenceMultiply(value, multiplier, shift);
    TINYMIND_FUZZ_CHECK(out == want);
    tinymind_fuzz::emitValue(out);

    checkRequantizer<int8_t>(value, multiplier, shift, want, br);
    checkRequantizer<uint8_t>(value, multiplier, shift, want, br);
    return 0;
}
