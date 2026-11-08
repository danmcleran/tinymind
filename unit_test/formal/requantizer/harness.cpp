/*
 * CBMC proofs for the int8 requantizer (cpp/qaffine.hpp), over every input
 * its contract allows:
 *
 *   TM_PROP_APPLY  Requantizer<int32_t, TM_DST>::apply is free of undefined
 *                  behaviour, and equals an independent int64 reference:
 *                  gemmlowp's two-step form (saturating left shift, Q0.31
 *                  doubling high multiply, rounding right shift), then the
 *                  zero point and the [qmin, qmax] clamp, all in int64. This
 *                  covers the one-rounding fast path and the +-2^30 rail of
 *                  the saturating path, which must both give that result.
 *   TM_PROP_MBQM   multiplyByQuantizedMultiplier (used directly by QAdd,
 *                  QGRU, QSSM, QCfC and others) is free of undefined
 *                  behaviour and equals the reference before the zero point.
 *   TM_PROP_LSHIFT the reference's left shift, written in the library's
 *                  shape so the solver can share the multiply, equals
 *                  acc * 2^left saturated to int32. Run once.
 *   TM_PROP_KERNELS the two 64-bit multiplies (saturatingRoundingDoubling-
 *                  HighMul, roundingHighMulDivideByPOT) cannot overflow for
 *                  any int32 operands, and the high multiply equals
 *                  gemmlowp's definition. Shift-independent; run once.
 *
 * Every accumulator, multiplier, zero point and clamp bound is
 * unconstrained. The shift is fixed per run (TM_SHIFT); the Makefile runs
 * every value in the contract, -31 .. 31 (see the README for the bounds).
 */

// CBMC 5.95's newest C++ front end is C++11; see arm_dsp_dot/harness.cpp for
// why declaring C++14 through _MSVC_LANG is sound here.
#define _MSVC_LANG 201402L
#include "include/tinymind_platform.hpp"
#include <cstdint>

#include "qaffine.hpp"

#ifndef TM_DST
#define TM_DST int8_t
#endif
#if !defined(TM_SHIFT) && !defined(TM_PROP_KERNELS) && !defined(TM_PROP_LSHIFT)
#error "define TM_SHIFT (the Makefile runs every value from -31 to 31)"
#endif

typedef TM_DST Dst;

int32_t nondet_int32();

// --- Reference, written from the gemmlowp definitions in int64 -------

static int64_t sat32(int64_t v)
{
    return (v > INT32_MAX) ? INT32_MAX : ((v < INT32_MIN) ? INT32_MIN : v);
}

// acc * 2^left saturated to int32. |acc| <= 2^31 and left <= 31 keep the
// int64 product within 2^62; a larger left shift saturates any non-zero
// accumulator.
static int64_t refLeftShift(int32_t acc, int32_t left)
{
    if (left == 0)
    {
        return acc;
    }
    if (left > 31)
    {
        return (acc > 0) ? INT32_MAX : ((acc < 0) ? INT32_MIN : 0);
    }
    // Shift in uint64: a left shift of a negative signed value is UB, and
    // this is the reference, not the code under test.
    return sat32(static_cast<int64_t>(static_cast<uint64_t>(static_cast<int64_t>(acc)) << left));
}

// The same saturating left shift, in the shape multiplyByQuantizedMultiplier
// writes it: a range test, then an int32 multiply by 2^left. The apply and
// mbqm proofs use this one, so that the library's 64-bit multiply and the
// reference's take syntactically the same operand and the solver can share
// them; with refLeftShift instead, it must prove two multipliers equivalent,
// which runs for hours per shift value. TM_PROP_LSHIFT proves this function
// equal to refLeftShift for every accumulator and shift, so the reference
// stays independent of the library by composition.
static int64_t refLeftShiftShaped(int32_t acc, int32_t left)
{
    if (left > 30)
    {
        return (acc > 0) ? INT32_MAX : ((acc < 0) ? INT32_MIN : 0);
    }
    if (left > 0)
    {
        const int32_t limit = INT32_MAX >> left;
        if (acc > limit)
        {
            return INT32_MAX;
        }
        if (acc < -limit - 1)
        {
            return INT32_MIN;
        }
        return acc * (static_cast<int32_t>(1) << left);
    }
    return acc;
}

// gemmlowp SaturatingRoundingDoublingHighMul, as gemmlowp writes it: nudge
// by 2^30 for a non-negative product and by 1 - 2^30 for a negative one, then
// divide by 2^31 truncating toward zero; INT32_MIN * INT32_MIN saturates.
// Deliberately not the library's one-add floor form, so the proof checks
// that derivation rather than restating it. Truncation is written with
// shifts of non-negative values only, which keeps the solver off a 64-bit
// divider.
static int64_t refHighMul(int64_t a, int64_t b)
{
    if ((a == INT32_MIN) && (b == INT32_MIN))
    {
        return INT32_MAX;
    }
    const int64_t ab = a * b;
    const int64_t nudge = (ab >= 0) ? (static_cast<int64_t>(1) << 30) : (1 - (static_cast<int64_t>(1) << 30));
    const int64_t n = ab + nudge;
    return (n >= 0) ? (n >> 31) : -((-n) >> 31);
}

// gemmlowp RoundingDivideByPOT: round to nearest, ties away from zero,
// i.e. floor((x + 2^(e-1) - [x < 0]) / 2^e).
static int64_t refRoundingDivide(int64_t x, int32_t e)
{
    if (e <= 0)
    {
        return x;
    }
    return (x + (static_cast<int64_t>(1) << (e - 1)) - ((x < 0) ? 1 : 0)) >> e;
}

static int64_t refScaled(int32_t acc, int32_t multiplier, int32_t shift)
{
    const int32_t left = (shift < 0) ? -shift : 0;
    const int32_t right = (shift > 0) ? shift : 0;
    return refRoundingDivide(refHighMul(refLeftShiftShaped(acc, left), multiplier), right);
}

static int64_t refApply(int32_t acc, int32_t multiplier, int32_t shift, int64_t zp, int64_t qmin, int64_t qmax)
{
    int64_t v = refScaled(acc, multiplier, shift) + zp;
    // Same clamp order as Requantizer::apply: low bound, then high.
    if (v < qmin)
    {
        v = qmin;
    }
    if (v > qmax)
    {
        v = qmax;
    }
    return v;
}

// Any value of the storage type (narrowing an unconstrained int32 reaches
// every one).
static Dst nondetDst()
{
    return static_cast<Dst>(nondet_int32());
}

// One proof per shift value (TM_SHIFT): with the shift a constant the solver
// faces only the multiply. The Makefile runs every value in the contract.
static int32_t nondetShift()
{
#if defined(TM_SHIFT)
    return TM_SHIFT;
#else
    return 0;
#endif
}

int main()
{
    tinymind::Requantizer<int32_t, Dst> rq;
    rq.multiplier = nondet_int32();
    rq.shift = nondetShift();
    rq.zero_point = nondetDst();
    rq.qmin = nondetDst();
    rq.qmax = nondetDst();
    const int32_t acc = nondet_int32();

#if defined(TM_PROP_APPLY)
    const Dst got = rq.apply(acc);
    const int64_t want = refApply(acc, rq.multiplier, rq.shift, rq.zero_point, rq.qmin, rq.qmax);
    __CPROVER_assert(static_cast<int64_t>(got) == want, "Requantizer::apply equals the int64 reference");
#elif defined(TM_PROP_MBQM)
    const int32_t got = tinymind::multiplyByQuantizedMultiplier(acc, rq.multiplier, rq.shift);
    __CPROVER_assert(static_cast<int64_t>(got) == refScaled(acc, rq.multiplier, rq.shift),
                     "multiplyByQuantizedMultiplier equals the int64 reference");
#elif defined(TM_PROP_KERNELS)
    // The 64-bit multiplies, on unconstrained arguments: every caller's
    // operands are int32 values, so these instances cover every call site.
    // The per-shift runs skip the multiply overflow checks, which are the
    // expensive ones, on the strength of this proof.
    const int32_t a = nondet_int32();
    const int32_t b = nondet_int32();
    const int32_t e = nondet_int32();
    __CPROVER_assume((e >= 2) && (e <= 31));
    __CPROVER_assert(static_cast<int64_t>(tinymind::saturatingRoundingDoublingHighMul(a, b)) == refHighMul(a, b),
                     "saturatingRoundingDoublingHighMul equals gemmlowp's definition");
    const int32_t r = tinymind::roundingHighMulDivideByPOT(a, b, e);
    (void)r;
#elif defined(TM_PROP_LSHIFT)
    // The shaped left shift equals the independent definition (acc * 2^left
    // in uint64, saturated to int32), for every accumulator and every left
    // shift a contract shift can produce, and a few beyond.
    const int32_t left = nondet_int32();
    __CPROVER_assume((left >= 0) && (left <= 40));
    __CPROVER_assert(refLeftShiftShaped(acc, left) == refLeftShift(acc, left),
                     "shaped left shift equals acc * 2^left saturated");
#else
#error "define TM_PROP_APPLY, TM_PROP_MBQM, TM_PROP_KERNELS or TM_PROP_LSHIFT"
#endif
    return 0;
}
