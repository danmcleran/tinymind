# cpp/include/simd

One header per ISA capability, dispatched by `simd_dispatch.hpp`. Repo-wide rules
are in the root `AGENTS.md`; the design (bit-exactness, non-goals, OpenMP) is in
`docs/ARCHITECTURE.md` under "SIMD Performance Backend".

## Gates

- `TINYMIND_ENABLE_SIMD_NEON` / `_NEON_DOTPROD` / `_NEON_FP16` / `_SVE` / `_SVE2` / `_HELIUM_MVE_I` / `_HELIUM_MVE_F` / `_ARM_DSP` / `_AVX2` / `_AVX_VNNI` / `_AVX512F` / `_AVX512_VNNI` — Phase 14 ISA-capability SIMD gates. Gates name ISA extensions, never CPU models. Headers in `cpp/include/simd/` open with a `static_assert` enforcing Arm's prerequisite chain (DOTPROD requires NEON; SVE/SVE2/FP16 vector forms require NEON; AVX_VNNI requires AVX2; AVX512_VNNI requires AVX512F; Helium MVE_I/MVE_F are M-profile only and mutually exclusive with NEON/SVE; ARM_DSP `#error`s unless the target defines `__ARM_FEATURE_DSP`, so it is refused on M0/M0+/M3/M23 and AArch64). Scalar fallback is byte-identical to the pre-Phase-14 build when every gate is off.

## Validation

- `check-simd-arm` (HARD, CI job "Arm SIMD backends (aarch64)" on `ubuntu-24.04-arm`, needs an aarch64 host) runs the A-profile backends no x86 host can execute: NEON, NEON dot product, NEON fp16 and SVE.
- Every compiled-in backend is also called directly from `unit_test/exhaustive` (every lane, every length 0..300 at every offset 0..15), because dispatch reaches only one per build.
- SVE is first in the Arm dispatch order (its gate requires the NEON gate, so with NEON first it was unreachable until #243), so `check-simd-arm-sve-native` runs the int8 suites with layers dispatched to SVE, and fails on a host without SVE unless `SVE_NATIVE=0`.
- Only the dot-product cases run under ASan+UBSan on arm64 (`BOOST_TEST_RUN_FILTERS`; Boost filters need the `test_suite_exhaustive/` prefix): the LUT case under ASan there does not finish in 15 min against 38 s on x86.
- SVE also runs under `qemu-aarch64 -cpu max,sve-default-vector-length=N` at 128/256/512/2048 bits (exhaustive dispatched + direct cases and the quantization suite), since SVE code is vector-length agnostic.
- Each target greps the suite's log for the backend it meant to run, so a silent scalar fallback fails; the job was shown to fail on three planted defects (a dropped tail element, an SVE stride fixed at 16 bytes, NEON switched off).
- `check-clang-avx2` (HARD) builds and runs the `TSAN_SUITES` dispatch consumers with the AVX2 backend compiled in; plain `check-clang` never reaches the hand-written intrinsics.
- The Arm DSP (SMLAD) and Helium backends cannot run on any CI host; they run on QEMU through `arm_dsp_qemu` and `helium_qemu` in `unit_test/embedded` (see its `AGENTS.md`), and the SMLAD dot product has a CBMC proof in `unit_test/formal`.

A new backend needs: its gate in `tinymind_platform.hpp` (default 0), a prerequisite `static_assert` with a compile-failure case in `simd_prereq_regressions`, a direct-call case in `unit_test/exhaustive`, and a CI target that greps the log for the backend name and was shown to fail on a planted defect.
