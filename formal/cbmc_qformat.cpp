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

// === Proof harnesses =========================================================

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
