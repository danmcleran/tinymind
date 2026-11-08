# Formal proof: the int8 requantizer

CBMC proofs that the requantizer in `cpp/qaffine.hpp` is free of undefined
behaviour and computes exactly what its definition says, for every input its
contract allows. Testing cannot cover this domain: a requantizer has a 32-bit
accumulator, a 32-bit multiplier, a shift and three storage-typed parameters,
and the unit and exhaustive suites sweep the accumulator for a handful of
fixed parameter sets.

```
PATH=$PWD/../arm_dsp_dot/tools:$PWD/../arm_dsp_dot/.venv/bin:$PATH make -j16
```

Bitwuzla setup is described in [`../arm_dsp_dot/README.md`](../arm_dsp_dot/README.md).
The 317 proofs take about 11 CPU-hours of solver time (50 minutes at `-j16`
on a 16-core desktop; the int32-storage jobs are the slowest, up to 20
minutes each). Each must print `PROVED`, then every planted defect must print
`CAUGHT`. That cost is why it is opt-in rather than a CI gate.

## What is proved

The harness includes the real header. Every accumulator, multiplier, zero
point, `qmin` and `qmax` is unconstrained (any value of its type); the shift
takes each value of the contract, -31 to 31, in its own job.

| Job | Statement |
|---|---|
| `apply_<type>_<shift>` | `Requantizer<int32_t, T>::apply` has no signed overflow and no undefined shift, and returns the reference result, for `T` = `int8_t`, `uint8_t`, `int16_t`, `int32_t` (252 jobs) |
| `mbqm_<shift>` | `multiplyByQuantizedMultiplier`, which QAdd, QGRU, QSSM, QCfC and others call directly, likewise against the reference before the zero point (63 jobs) |
| `kernels` | The two 64-bit multiplies cannot overflow for any int32 operands, and `saturatingRoundingDoublingHighMul` equals gemmlowp's definition |
| `lshift` | The reference's left shift, written in the library's shape (see below), equals `acc * 2^left` computed in uint64 and saturated to int32, for every accumulator and every left shift 0..40 |

The **reference** is written independently from the gemmlowp definitions and
computed in int64, where nothing can overflow: the accumulator left-shifted
and saturated to int32, then gemmlowp's `SaturatingRoundingDoublingHighMul`
in its original form (a sign-dependent nudge and a division truncating toward
zero, not the library's one-add floor), then `RoundingDivideByPOT` as
`floor((x + 2^(e-1) - [x < 0]) / 2^e)`, not the library's mask and threshold,
then the zero point and the `[qmin, qmax]` clamp. `apply` has three code
paths (the one-rounding path for a scale ratio below 1/4, the saturating
path, and the int64 tail for 32-bit storage); all three must equal that one
reference, so the proof also covers the derivations the header's comments
argue for: the single 64-bit rounding, and the +-2^30 rail that replaced an
exact window on Cortex-M0+.

Two things keep the per-shift jobs fast. First, the reference's left shift is
written in the same shape as the library's (a range test, then an int32
multiply by 2^left), so the library's 64-bit multiply and the reference's
take syntactically the same operand and the solver can share them. Written
independently, the two operands are equal but different terms, and the solver
has to prove two 64-bit multipliers equivalent: over 12 minutes per shift
value, against 20 seconds. The `lshift` job proves the shaped version equal to
the independent one, so the reference is still independent of the library by
composition. Second, the per-shift jobs leave out the three 64-bit multiply overflow checks (two in
the library, one in the reference). Those are by far the most expensive
obligations, and the `kernels` job proves them for every pair of int32
operands, which covers every call site. `select_props.sh` makes the selection;
every other check CBMC generates is kept.

## Found by this proof

Both are fixed in the same change, with regression tests in
`unit_test/quantization`:

- **A scale ratio below 2^-32 gave undefined behaviour.** `quantizeMultiplier`
  returned a shift of 32 or more for it, and `apply` then shifted a 32-bit
  value by that much. Such a ratio rounds every int32 accumulator to 0, so
  `quantizeMultiplier` now returns multiplier 0, and clamps the left-shift
  side at -31 (beyond which everything saturates anyway). The shift contract,
  -31 to 31, is now documented on `Requantizer`.
- **32-bit storage overflowed on the one-rounding path.** The saturating path
  clamped to the exact `[qmin - zp, qmax - zp]` window before adding a 32-bit
  zero point, but the one-rounding path added it in int32 unclamped. 32-bit
  storage now adds and clamps in int64 on both paths. The int8, uint8 and
  int16 instantiations compile to identical machine code before and after
  (Cortex-M0+ and Cortex-M4, `-O3`), so no benchmark moves.

## Planted defects

Each is applied to a copy of the header and run at one point where it must
be reported; `make mutants` fails if any survives, or if CBMC reaches no
verdict at all.

| Mutant | What it breaks |
|---|---|
| `threshold` | rounding threshold of `roundingDivideByPOT` loses its negative-value adjustment |
| `nudge` | the one-rounding path's 2^30 nudge halved |
| `rail` | the saturating path's rail at 2^14, inside int16's range |
| `int32_zp` | the 32-bit-storage fix undone |
| `left_limit` | the saturating left shift's overflow limit off by one bit |

## Scope and trusted base

- **CBMC's semantics are trusted**, notably that `>>` of a negative value is
  arithmetic (implementation-defined before C++20; GCC and Clang agree) and
  that narrowing conversions wrap. The proof checks undefined behaviour, not
  implementation-defined behaviour.
- **`FastRequantizer` is not covered.** CBMC 5.95's C++ front end cannot
  type-check it (a member function template, a member typedef used as a
  scope, and a static member function of an explicit specialization all
  fail), and restructuring that inner-loop code to suit the tool was not
  worth the risk. The exhaustive suite compares it against `Requantizer`.
- **Storage types:** `int8_t`, `uint8_t`, `int16_t` and `int32_t`, which
  cover every instantiation in the tree. `uint16_t` shares the narrow-storage
  path with `int16_t`.
- **`SrcAccum` is `int32_t`**, the only accumulator type in use; `apply`
  narrows any other to int32 first.
- Like the other proof here, the harness defines `_MSVC_LANG` to get past the
  C++14 floor with CBMC's C++11 front end; see `../arm_dsp_dot/harness.cpp`.
