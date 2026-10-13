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

// AdditionOp branch (qformat.hpp). T = int16_t is signed, so the policy's
// unsigned-only "target > maxValue" shortcut is inactive here.
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

// DivisionOp branch (qformat.hpp): divisor==0 handled first (the bound for
// the numerator's sign, 0 for 0 / 0).
static int32_t sat_div_i32(int32_t o, int32_t t, int32_t lo, int32_t hi)
{
    int32_t result = 0;
    if (t == 0)            { result = (o > 0) ? hi : ((o < 0) ? lo : 0); }
    else if ((o / t) > hi) { result = hi; }
    else if ((o / t) < lo) { result = lo; }
    else                   { result = o / t; }
    return result;
}

// WrapPolicy<T>::saturate DivisionOp branch (qformat.hpp): divisor==0 gives
// the bound for the numerator's sign and 0 for 0 / 0, as MinMaxSaturatePolicy
// does.
static int32_t wrap_div_i32(int32_t o, int32_t t, int32_t lo, int32_t hi)
{
    int32_t result = 0;
    if (t == 0) { result = (o > 0) ? hi : ((o < 0) ? lo : 0); }
    else        { result = o / t; }
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

// === Transcriptions of the QValue storage arithmetic (32-bit word) ==========
// FullWidthValueType = int32_t, FullWidthFieldType = uint32_t; the format is
// passed as (fixedBits, fracBits). Proved for Q8.16, the narrow shape issue
// #187 was about (24 bits in the word, so bits 24..31 must track the sign), and
// for Q1.31, where the format fills the word with one fixed bit and 2^31 is
// not representable in the signed word -- the case a signed multiply in
// fromParts overflowed on.

// int32_t -> uint32_t as C++ [conv.integral] defines it: the value modulo 2^32.
// The shipping code writes this as a plain static_cast, which is well-defined;
// CBMC's --conversion-check flags every negative-to-unsigned cast regardless,
// so the modular result is spelled out here to keep that check meaningful for
// the narrowings that must stay in range.
static uint32_t as_u32(int32_t value)
{
    return (value < 0) ? ((uint32_t)(value + 1 + INT32_MAX) + 0x80000000u) : (uint32_t)value;
}

// uint32_t -> int32_t modulo 2^32, the conversion QValue relies on when it
// reinterprets an unsigned word (modular on GCC and Clang, and defined so by
// C++20). Spelled out for the same reason as as_u32.
static int32_t as_i32(uint32_t value)
{
    return (value > (uint32_t)INT32_MAX) ? ((int32_t)(value - 0x80000000u) - INT32_MAX - 1) : (int32_t)value;
}

// tinymind::FormatWidthNormalizer<int32_t, uint32_t, numBits, true>::normalize
// (the FillsWord specialization is the identity).
static int32_t normalize_i32(int32_t value, unsigned numBits)
{
    if (numBits == 32)
    {
        return value;
    }

    const uint32_t mask = (uint32_t)(((uint32_t)1 << numBits) - 1);
    const uint32_t signBit = (uint32_t)((uint32_t)1 << (numBits - 1));
    const uint32_t low = (uint32_t)(as_u32(value) & mask);
    return (int32_t)((int32_t)(low ^ signBit) - (int32_t)signBit);
}

// QValue::fractionalPartMask
static uint32_t frac_mask(unsigned fracBits)
{
    return (uint32_t)(((uint32_t)1 << fracBits) - 1);
}

// QValue::fromParts: each part reduced to its own width, joined in the
// unsigned word, reinterpreted, normalized to the format's width.
static int32_t from_parts(int32_t fixedPart, uint32_t fractionalPart, unsigned fixedBits, unsigned fracBits)
{
    const uint32_t fixedWord = as_u32(normalize_i32(fixedPart, fixedBits));
    const uint32_t fractionalWord = (uint32_t)(fractionalPart & frac_mask(fracBits));
    const uint32_t word = (uint32_t)((uint32_t)(fixedWord << fracBits) | fractionalWord);
    return normalize_i32(as_i32(word), fixedBits + fracBits);
}

// QValue::getFixedPart
static int32_t fixed_part(int32_t value, unsigned fixedBits, unsigned fracBits)
{
    const int32_t shifted = (int32_t)(as_u32(value) >> fracBits);
    return normalize_i32(shifted, fixedBits);
}

// QValue::getFractionalPart
static uint32_t frac_part(int32_t value, unsigned fracBits)
{
    return (uint32_t)(as_u32(value) & frac_mask(fracBits));
}

// The three storage properties for one format: any word normalizes into
// range, building from ANY parts (even out-of-range ones, which truncate as the
// old bit-field store did) is overflow-free and canonical with in-range parts
// reading back exactly, and every canonical value round-trips through its
// parts.
static void prove_storage(unsigned fixedBits, unsigned fracBits)
{
    const unsigned width = fixedBits + fracBits;
    const int64_t lo = -((int64_t)1 << (width - 1));
    const int64_t hi = ((int64_t)1 << (width - 1)) - 1;
    const int32_t fixedLo = (int32_t)(-((int64_t)1 << (fixedBits - 1)));
    const int32_t fixedHi = (int32_t)(((int64_t)1 << (fixedBits - 1)) - 1);

    int32_t v;
    const int32_t r = normalize_i32(v, width);
    __CPROVER_assert(r >= lo && r <= hi, "normalized value is in the format's range");
    __CPROVER_assert((width == 32) || ((as_u32(r) & (((uint32_t)1 << width) - 1)) == (as_u32(v) & (((uint32_t)1 << width) - 1))), "normalize keeps the format's bits");
    __CPROVER_assert(normalize_i32(r, width) == r, "normalize is idempotent");

    int32_t fixedPart;
    uint32_t fractionalPart;
    const int32_t built = from_parts(fixedPart, fractionalPart, fixedBits, fracBits);
    __CPROVER_assert(built >= lo && built <= hi, "fromParts stays in the format's range");
    __CPROVER_assert(normalize_i32(built, width) == built, "fromParts yields a canonical value");

    if ((fixedPart >= fixedLo) && (fixedPart <= fixedHi) && (fractionalPart <= frac_mask(fracBits)))
    {
        __CPROVER_assert(fixed_part(built, fixedBits, fracBits) == fixedPart, "fixed part reads back");
        __CPROVER_assert(frac_part(built, fracBits) == fractionalPart, "fractional part reads back");
    }

    int32_t canonical;
    __CPROVER_assume(canonical >= lo && canonical <= hi);
    const int32_t rebuilt = from_parts(fixed_part(canonical, fixedBits, fracBits), frac_part(canonical, fracBits), fixedBits, fracBits);
    __CPROVER_assert(rebuilt == canonical, "parts round-trip to the same value");
}

// === Proof harnesses =========================================================

extern "C" void prove_storage_q8_16()
{
    prove_storage(8, 16);
}

extern "C" void prove_storage_q1_31()
{
    prove_storage(1, 31);
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
    __CPROVER_assume(lo <= 0 && 0 <= hi); // a QValue range always contains 0, which 0 / 0 returns

    const int32_t r = sat_div_i32(o, t, lo, hi);

    __CPROVER_assert(r >= lo && r <= hi, "div saturation stays in [lo,hi]");
}

// QValue<8,8,true,*,WrapPolicy>::operator/=: the dividend is the Q8.8 word
// scaled by 2^8 in the int32_t division type, the divisor is any Q8.8 word,
// zero included. --div-by-zero-check and --signed-overflow-check make the
// absence of undefined behaviour the proof obligation.
extern "C" void prove_wrap_div_q8_8()
{
    int32_t o, t;
    __CPROVER_assume(o >= Q8_8_MIN && o <= Q8_8_MAX);
    __CPROVER_assume(t >= Q8_8_MIN && t <= Q8_8_MAX);

    const int32_t r = wrap_div_i32(o * 256, t, Q8_8_MIN, Q8_8_MAX);

    __CPROVER_assert((t != 0) || (r == ((o > 0) ? Q8_8_MAX : ((o < 0) ? Q8_8_MIN : 0))),
                     "division by zero gives the bound for the numerator's sign, and 0 for 0 / 0");
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
