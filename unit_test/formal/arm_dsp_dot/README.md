# Formal equivalence proof: SMLAD dot product vs scalar reference

Pilot. Uses [CBMC](https://www.cprover.org/cbmc/) to prove that
`tinymind::simd::arm_dsp::int8DotWithZeroPoint` (`cpp/include/simd/simd_arm_dsp.hpp`)
returns the same `int32_t` as `int8DotWithZeroPointScalar`
(`cpp/include/simd/simd_dispatch.hpp`) for **every** value of `x`, `w` and `zp`.

The harness includes the real headers. Nothing from the library is copied
into the harness.

## What is proved

For each length `n` in 0..8, with all input bytes and the zero point fully
symbolic (2^(16n+8) inputs per length), CBMC proves:

- the SMLAD result equals the scalar reference;
- the SXTAB16 lanes never wrap (the header's "`x - zp` lies in [-255, 255]"
  claim), and SMLAD never sets the Q flag;
- no out-of-bounds or invalid pointer access, no signed overflow, and no
  undefined shift, in either implementation;
- loop unwinding is complete (`--unwinding-assertions`).

Further runs repeat the proof with misaligned buffers and with a big-endian
memory model. Those confirm the header's claim that its byte pairing does not
depend on load endianness.

## Extending to every `n`

The loop is `acc <- SMLAD(.., SMLAD(.., acc))` once per 4-byte block, then
a scalar tail. By SMLAD's definition, each block adds a value that depends
only on that block's 8 bytes and `zp`, so block `k` contributes the same
amount whatever `acc` holds when it starts. The `n = 4` proof shows that
contribution equals the four scalar products, for all inputs. The
`n = 0..3` proofs cover every tail. By induction on the number of blocks the
two implementations agree for every `n` with no int32 overflow. This step is
argued on paper, not checked by CBMC.

**Precondition:** `n <= 65793`. A term `w*(x - zp)` is at most 32640 in
magnitude, so the sum cannot overflow int32 below that length. Above it the
scalar reference's `acc +=` is signed overflow (undefined behaviour), while
SMLAD wraps and sets Q. Real layers are far below this bound.

## Trusted base (what is *not* proved)

- `shim/arm_acle.h`: the instruction semantics, transcribed by hand from the
  Armv7-M ARM pseudocode. The `arm_dsp_qemu` gate cross-checks it by running
  the real instructions.
- Only the **Clang** source path is verified. The GCC path spells the two
  rotates as inline `asm` (`sxtb16 ... ror #8`), which CBMC cannot read.
  Those two lines are checked by inspection and by `arm_dsp_qemu`.
- The compiler, the hardware, CBMC and the SMT solver.

## Planted defects

`make mutants` applies four single-line defects to a copy of the header, and
each must be reported as an equivalence failure: wrong byte pairing,
zero-point sign flip, wrong tail mask, and one-sided wrong rotate. During
development a two-sided wrong rotate *survived*. That mutant turned out to be
equivalent (lanes `(b3, b1)` on both operands still pair `w[i]` with `x[i]`),
so the prover was right and the mutant was replaced.

## Running

```
make -j16          # 15 proofs + 4 mutants, about 100 s on 16 cores
```

Needs `cbmc` (tested 5.95.1) and an SMT solver on `PATH`. Default
`SOLVER=--bitwuzla`; `SOLVER=--z3` works but is about 5x slower. The
`bitwuzla` PyPI package ships only Python bindings, so `tools/bitwuzla` is a
small command-line wrapper that CBMC can call:

```
python3 -m venv .venv && .venv/bin/pip install bitwuzla   # or z3-solver
PATH=$PWD/tools:$PWD/.venv/bin:$PATH make -j16
```

Run it with the venv's interpreter first on `PATH`, so the wrapper's
`#!/usr/bin/env python3` finds the bindings. CBMC's built-in
MiniSat cannot finish `n = 4` in reasonable time, because proving that two
multipliers are equivalent is the classic hard case for bit-level SAT.

## Pitfall found during the pilot

CBMC's C++ front end has no model for `__builtin_memcpy`. It skips the call
and prints only a warning, so `load4()` returned an unconstrained word and the
first run reported a **spurious counterexample**. The harness routes the
builtin to `std::memcpy`, which CBMC does model, and the Makefile now fails any
run whose log contains `no body for function`.
