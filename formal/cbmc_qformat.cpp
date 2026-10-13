/*
 * CBMC bounded-model-checking proofs for the qformat.hpp fixed-point kernels.
 *
 * Why a transcription: CBMC 5.x's built-in C++ frontend cannot instantiate the
 * templated policy methods in qformat.hpp (it aborts type-checking the QValue
 * template-template parameters). So this file transcribes the kernel arithmetic
 * concretely, preserving C++ integer-promotion semantics exactly, and proves
 * THAT. The shipping templates implement the identical operations and are
 * exercised by unit_test/qformat and unit_test/nn; this harness adds the
 * machine-checked guarantee that the arithmetic itself is free of signed
 * overflow, bad conversions and division UB, and that saturation stays in range
 * -- over ALL inputs in the modeled (Q8.8) domain, not just sampled ones.
 *
 * Each transcription is annotated with the qformat.hpp policy it mirrors. If a
 * policy changes, update the matching function here.
 *
 * Run via formal/Makefile (`make prove`).
 *
 * Copyright (c) Dan McLeran. MIT (see LICENSE.txt).
 */
#include <cstdint>

// Q8.8 domains: raw storage is int16; the multiply/divide accumulator is int32.
static const int Q8_8_MIN = -32768;
static const int Q8_8_MAX = 32767;
// Post-multiply accumulator bound (|o*t| <= 32767*32767).
static const int32_t ACC_BOUND = 32767 * 32767;

// === Transcriptions of tinymind::MinMaxSaturatePolicy<T>::saturate ===========
// For T=int16_t the operands promote to int for every guard/arith expression,
// then the chosen branch narrows back to int16_t on assignment -- modeled here
// with int parameters and an int16_t result.

// AdditionOp branch (qformat.hpp).
static int16_t sat_add_i16(int o, int t, int lo, int hi)
{
    int16_t result = 0;
    if (t > 0)      { result = (o > (hi - t)) ? (int16_t)hi : (int16_t)(o + t); }
    else if (t < 0) { result = (o < (lo - t)) ? (int16_t)lo : (int16_t)(o + t); }
    else            { result = (int16_t)o; }
    return result;
}

// SubtractionOp branch (qformat.hpp).
static int16_t sat_sub_i16(int o, int t, int lo, int hi)
{
    int16_t result = 0;
    if (t > 0)      { result = (o < (lo + t)) ? (int16_t)lo : (int16_t)(o - t); }
    else if (t < 0) { result = (o > (hi + t)) ? (int16_t)hi : (int16_t)(o - t); }
    else            { result = (int16_t)o; }
    return result;
}

// MultiplicationOp branch (qformat.hpp): product computed in T before clamping.
static int32_t sat_mul_i32(int32_t o, int32_t t, int32_t lo, int32_t hi)
{
    int32_t result = 0;
    if ((o * t) > hi)      { result = hi; }
    else if ((o * t) < lo) { result = lo; }
    else                   { result = o * t; }
    return result;
}

// DivisionOp branch (qformat.hpp): divisor==0 handled first.
static int32_t sat_div_i32(int32_t o, int32_t t, int32_t lo, int32_t hi)
{
    int32_t result = 0;
    if (t == 0)            { result = (o > 0) ? hi : lo; }
    else if ((o / t) > hi) { result = hi; }
    else if ((o / t) < lo) { result = lo; }
    else                   { result = o / t; }
    return result;
}

// === Transcriptions of the rounding policies (8 fractional bits) =============
// tinymind::RoundUpPolicy<T,Frac>::round
static int32_t round_up_i32(int32_t value)
{
    int32_t result = value;
    result = result + ((result & (1 << (8 - 1))) << 1);
    result >>= 8;
    return result;
}

// tinymind::TruncatePolicy<T,Frac>::round
static int32_t truncate_i32(int32_t value)
{
    int32_t result = value;
    result >>= 8;
    return result;
}

// === Transcriptions of the narrow-format storage arithmetic (Q8.16) ==========
// Q8.16 is 24 bits held in an int32_t (unsigned counterpart uint32_t), the
// shape issue #187 was about: every write must leave bits 24..31 equal to the
// sign bit. FullWidthValueType = int32_t, FullWidthFieldType = uint32_t.
static const int32_t Q8_16_MIN = -(1 << 23);
static const int32_t Q8_16_MAX = (1 << 23) - 1;

// int32_t -> uint32_t as C++ [conv.integral] defines it: the value modulo 2^32.
// The shipping code writes this as a plain static_cast, which is well-defined;
// CBMC's --conversion-check flags every negative-to-unsigned cast regardless,
// so the modular result is spelled out here to keep that check meaningful for
// the unsigned-to-signed narrowings, which must stay in range.
static uint32_t as_u32(int32_t value)
{
    return (value < 0) ? ((uint32_t)(value + 1 + INT32_MAX) + 0x80000000u) : (uint32_t)value;
}

// tinymind::FormatWidthNormalizer<ValueType, FieldType, NumBits, true>::normalize
// for NumBits = 24 (value) and NumBits = 8 (fixed part).
static int32_t normalize_i32(int32_t value, unsigned numBits)
{
    const uint32_t mask = (uint32_t)(((uint32_t)1 << numBits) - 1);
    const uint32_t signBit = (uint32_t)((uint32_t)1 << (numBits - 1));
    const uint32_t low = (uint32_t)(as_u32(value) & mask);
    return (int32_t)((int32_t)(low ^ signBit) - (int32_t)signBit);
}

// QValue::fractionalPartMask for 16 fractional bits.
static uint32_t frac_mask_q8_16()
{
    return (uint32_t)(((uint32_t)1 << 16) - 1);
}

// QValue::fromParts
static int32_t from_parts_q8_16(int32_t fixedPart, uint32_t fractionalPart)
{
    const int32_t fixedValue = normalize_i32(fixedPart, 8);
    const int32_t fractionalValue = (int32_t)(fractionalPart & frac_mask_q8_16());
    const int32_t scale = (int32_t)((int32_t)1 << 16);
    return (int32_t)((fixedValue * scale) + fractionalValue);
}

// QValue::getFixedPart
static int32_t fixed_part_q8_16(int32_t value)
{
    const int32_t shifted = (int32_t)(as_u32(value) >> 16);
    return normalize_i32(shifted, 8);
}

// QValue::getFractionalPart
static uint32_t frac_part_q8_16(int32_t value)
{
    return (uint32_t)(as_u32(value) & frac_mask_q8_16());
}

// === Proof harnesses =========================================================

// Any storage word normalizes to the in-range value with the same low 24 bits,
// and normalizing is idempotent -- so a canonical value is never disturbed.
extern "C" void prove_normalize_q8_16()
{
    int32_t v;
    const int32_t r = normalize_i32(v, 24);

    __CPROVER_assert(r >= Q8_16_MIN && r <= Q8_16_MAX, "normalized value is in the Q8.16 range");
    __CPROVER_assert((as_u32(r) & 0xFFFFFFu) == (as_u32(v) & 0xFFFFFFu), "normalize keeps the format's 24 bits");
    __CPROVER_assert(normalize_i32(r, 24) == r, "normalize is idempotent");
}

// Building from ANY parts (even out-of-range ones, which truncate as the old
// bit-field store did) is overflow-free and yields a canonical value; in-range
// parts read back exactly.
extern "C" void prove_from_parts_q8_16()
{
    int32_t fixedPart;
    uint32_t fractionalPart;
    const int32_t r = from_parts_q8_16(fixedPart, fractionalPart);

    __CPROVER_assert(r >= Q8_16_MIN && r <= Q8_16_MAX, "fromParts stays in the Q8.16 range");
    __CPROVER_assert(normalize_i32(r, 24) == r, "fromParts yields a canonical value");

    if ((fixedPart >= -128) && (fixedPart <= 127) && (fractionalPart <= 0xFFFFu))
    {
        __CPROVER_assert(fixed_part_q8_16(r) == fixedPart, "fixed part reads back");
        __CPROVER_assert(frac_part_q8_16(r) == fractionalPart, "fractional part reads back");
    }
}

// Every canonical value splits into parts that rebuild it exactly.
extern "C" void prove_parts_round_trip_q8_16()
{
    int32_t v;
    __CPROVER_assume(v >= Q8_16_MIN && v <= Q8_16_MAX);

    const int32_t rebuilt = from_parts_q8_16(fixed_part_q8_16(v), frac_part_q8_16(v));

    __CPROVER_assert(rebuilt == v, "parts round-trip to the same value");
}

extern "C" void prove_saturate_add_sub_int16()
{
    int o, t, lo, hi;
    __CPROVER_assume(lo >= Q8_8_MIN && lo <= Q8_8_MAX);
    __CPROVER_assume(hi >= Q8_8_MIN && hi <= Q8_8_MAX);
    __CPROVER_assume(t  >= Q8_8_MIN && t  <= Q8_8_MAX);
    __CPROVER_assume(lo <= hi);
    // Precondition the policy relies on: the accumulator is already a valid
    // in-range Q-format value. The add/sub branches clamp only the bound they
    // move toward (t>0 checks hi, t<0 checks lo), which is sound exactly
    // because origValue starts within [lo,hi] -- the invariant QValue keeps.
    __CPROVER_assume(o >= lo && o <= hi);

    const int16_t ra = sat_add_i16(o, t, lo, hi);
    const int16_t rs = sat_sub_i16(o, t, lo, hi);

    __CPROVER_assert(ra >= lo && ra <= hi, "add saturation stays in [lo,hi]");
    __CPROVER_assert(rs >= lo && rs <= hi, "sub saturation stays in [lo,hi]");
}

extern "C" void prove_saturate_mul_q8_8()
{
    int32_t o, t, lo, hi;
    __CPROVER_assume(o >= Q8_8_MIN && o <= Q8_8_MAX);
    __CPROVER_assume(t >= Q8_8_MIN && t <= Q8_8_MAX);
    __CPROVER_assume(lo <= hi);

    const int32_t r = sat_mul_i32(o, t, lo, hi);

    __CPROVER_assert(r >= lo && r <= hi, "mul saturation stays in [lo,hi]");
}

extern "C" void prove_saturate_div_q8_8()
{
    int32_t o, t, lo, hi;
    __CPROVER_assume(o >= Q8_8_MIN && o <= Q8_8_MAX);
    __CPROVER_assume(t >= Q8_8_MIN && t <= Q8_8_MAX);
    __CPROVER_assume(lo <= hi);

    const int32_t r = sat_div_i32(o, t, lo, hi);

    __CPROVER_assert(r >= lo && r <= hi, "div saturation stays in [lo,hi]");
}

extern "C" void prove_round_up_q8_8()
{
    int32_t v;
    __CPROVER_assume(v >= -ACC_BOUND && v <= ACC_BOUND);
    volatile int32_t r = round_up_i32(v);
    (void)r;
}

extern "C" void prove_truncate_q8_8()
{
    int32_t v;
    __CPROVER_assume(v >= -ACC_BOUND && v <= ACC_BOUND);
    volatile int32_t r = truncate_i32(v);
    (void)r;
}
