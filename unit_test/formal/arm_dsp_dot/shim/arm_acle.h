/*
 * Bit-precise host model of the four ACLE DSP intrinsics simd_arm_dsp.hpp
 * uses, transcribed from the Arm Architecture Reference Manual (Armv7-M,
 * DDI 0403) pseudocode for SXTB16, SXTAB16 and SMLAD. This model is part of
 * the trusted base of the proof: CBMC verifies the algorithm against these
 * definitions, not against silicon. The QEMU gate (arm_dsp_qemu) covers the
 * gap by running the real instructions.
 *
 * Each model computes exactly what the hardware writes, including wraparound,
 * and additionally asserts that no wrap happens. The hardware does not trap on
 * a wrap; the asserts check the header's claim that its operand ranges never
 * reach one.
 */
#pragma once

#include <stdint.h>

typedef int32_t int8x4_t;
typedef int32_t int16x2_t;

static inline uint32_t tm_lane16(int32_t lo, int32_t hi)
{
    return (static_cast<uint32_t>(lo) & 0xFFFFu) |
           ((static_cast<uint32_t>(hi) & 0xFFFFu) << 16);
}

static inline int32_t tm_slane16(uint32_t v, unsigned lane)
{
    return static_cast<int16_t>(static_cast<uint16_t>(v >> (16u * lane)));
}

static inline int32_t tm_sbyte(uint32_t v, unsigned byte)
{
    return static_cast<int8_t>(static_cast<uint8_t>(v >> (8u * byte)));
}

// SXTB16 Rd, Rm  (rotation 0):
//   Rd<15:0>  = SignExtend(Rm<7:0>, 16)
//   Rd<31:16> = SignExtend(Rm<23:16>, 16)
static inline int16x2_t __sxtb16(int8x4_t m)
{
    const uint32_t u = static_cast<uint32_t>(m);
    return static_cast<int16x2_t>(tm_lane16(tm_sbyte(u, 0), tm_sbyte(u, 2)));
}

// SXTAB16 Rd, Rn, Rm  (rotation 0):
//   Rd<15:0>  = Rn<15:0>  + SignExtend(Rm<7:0>, 16)     (mod 2^16)
//   Rd<31:16> = Rn<31:16> + SignExtend(Rm<23:16>, 16)   (mod 2^16)
static inline int16x2_t __sxtab16(int16x2_t n, int8x4_t m)
{
    const uint32_t un = static_cast<uint32_t>(n);
    const uint32_t um = static_cast<uint32_t>(m);
    const int32_t lo = tm_slane16(un, 0) + tm_sbyte(um, 0);
    const int32_t hi = tm_slane16(un, 1) + tm_sbyte(um, 2);
    __CPROVER_assert(lo >= INT16_MIN && lo <= INT16_MAX, "SXTAB16 low lane does not wrap");
    __CPROVER_assert(hi >= INT16_MIN && hi <= INT16_MAX, "SXTAB16 high lane does not wrap");
    return static_cast<int16x2_t>(tm_lane16(lo, hi));
}

// SMLAD Rd, Rn, Rm, Ra:
//   product1 = SInt(Rn<15:0>)  * SInt(Rm<15:0>)
//   product2 = SInt(Rn<31:16>) * SInt(Rm<31:16>)
//   result   = product1 + product2 + SInt(Ra)
//   Rd       = result<31:0>;  Q flag set if result != SInt(result<31:0>)
static inline int32_t __smlad(int16x2_t n, int16x2_t m, int32_t a)
{
    const uint32_t un = static_cast<uint32_t>(n);
    const uint32_t um = static_cast<uint32_t>(m);
    const int64_t result = static_cast<int64_t>(tm_slane16(un, 0)) * tm_slane16(um, 0) +
                           static_cast<int64_t>(tm_slane16(un, 1)) * tm_slane16(um, 1) +
                           static_cast<int64_t>(a);
    __CPROVER_assert(result >= INT32_MIN && result <= INT32_MAX, "SMLAD does not saturate (Q flag clear)");
    return static_cast<int32_t>(static_cast<uint32_t>(static_cast<uint64_t>(result)));
}
