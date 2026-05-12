# Quantization Plan for TinyMind

Post-training int8 quantization (TFLite-style: per-tensor/per-channel scale + zero-point + int32 accumulators + integer requantization). Additive only — does not modify existing Q-format or float layers.

## Background: Q-format vs. quantization

These are distinct:

- **Q-format substitution** (current TinyMind): swap representation. Float `0.7531` → Q8.8 `0.7539` (nearest grid point). Resolution set by fractional bits. No calibration needed — just type swap. `QValue` carries no runtime metadata; `ValueType` is a single template parameter threaded through every layer.
- **Quantization** (this plan): map continuous distribution to discrete int8 levels via calibrated `scale` + `zero_point`. `q = round(x / scale) + zero_point`. Scale fit per tensor (or per channel) from observed min/max. Requires runtime metadata, requantization between layers, and int32 accumulators paired with int8 weights/activations.

## Design constraints

- Existing layers (`conv1d.hpp`, `conv2d.hpp`, `pool*.hpp`, `batchnorm.hpp`, `dropout.hpp`, `selfattention1d.hpp`, `fft1d.hpp`, `binarylayer.hpp`, `ternarylayer.hpp`, `neuralnet.hpp`) are all single-`ValueType` templates. **Do not modify them.**
- `QValue` (`qformat.hpp`) has no runtime scale/zero-point and no accumulator-type separation. **Do not modify it.**
- Add a parallel quantization path. User opts in via type choice + new feature gate.
- Preserve the four-corner `(FLOAT, STD)` embedded regression matrix; add a fifth axis `QUANT`.

## New feature gate

In `cpp/include/tinymind_platform.hpp`:

- `TINYMIND_ENABLE_QUANTIZATION` (default `0`) — gates new quantization headers entirely. Embedded freestanding builds untouched unless flipped.

Inference-only quantized build target: `TINYMIND_ENABLE_QUANTIZATION=1`, `TINYMIND_ENABLE_FLOAT=0`, `TINYMIND_ENABLE_STD=0`. Calibration tooling lives behind `TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD` so host-only.

## Phase 1 — Core types (`cpp/qaffine.hpp`, new file)

```cpp
template<typename StorageType, typename AccumulatorType>
struct QAffineTensor {
    // runtime metadata, per-tensor (extend later to per-channel)
    float scale;
    StorageType zero_point;
    // raw storage held by caller (layer owns weight buffers)
};

template<typename SrcAccum, typename DstStorage>
struct Requantizer {
    int32_t multiplier;   // fixed-point M0 (Q0.31)
    int8_t  shift;        // right shift after multiply
    DstStorage zero_point;
    static DstStorage apply(SrcAccum acc);  // saturating round
};
```

- int8 activations + int8 weights + int32 accumulator (TFLite reference kernel shape).
- Requantization done as int multiply + shift (no float in inference path) — matches CMSIS-NN style.
- Calibration helpers (float → scale/zp + multiplier/shift decomposition) live behind `TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD`. Inference-only build gets no float pull-in.

## Phase 2 — Quantized layers (parallel to existing)

New headers, mirror existing API but template on `<InputType, WeightType, AccumType, OutputType>`:

- `cpp/qconv2d.hpp` — `QConv2D` with int32 accumulator, requantize on output
- `cpp/qdepthwiseconv2d.hpp` — `QDepthwiseConv2D` (per-channel scale critical here — TFLite mandates)
- `cpp/qpointwiseconv2d.hpp` — `QPointwiseConv2D`
- `cpp/qpool2d.hpp` — `QMaxPool2D` / `QAvgPool2D` (avg needs requantize, max is rescale-free if same scale)
- `cpp/qdense.hpp` — quantized fully-connected (standalone, not coupled to `NeuralNet<>`)

Each layer carries its own `Requantizer` in member state. No changes to non-`Q*` siblings.

## Phase 3 — Weight import path

Leave `ValueConverter` in `cpp/include/nnproperties.hpp` alone. Add separate:

- `cpp/include/qcalibration.hpp` — float → int8 quantization with per-tensor or per-channel scale fitting (min/max + symmetric/asymmetric). Host-only (`TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD`).
- Helper to compute requant `(multiplier, shift)` from `(input_scale * weight_scale) / output_scale`.

PyTorch import path in `examples/pytorch/` gets a sibling: `examples/pytorch_quant/` showing quantize-aware weight export. Existing example untouched.

## Phase 4 — Activation functions

- Quantized ReLU = clamp at zero-point.
- Quantized sigmoid/tanh = lookup table (similar pattern to `lookupTables.cpp` but indexed by int8 input).
- Add `cpp/qactivations.hpp`. Existing `activationFunctions.hpp` untouched.

## Phase 5 — Tests + bench

- `unit_test/quantization/` — new Boost.Test suite. Verify:
  - `QConv2D` matches float reference within tolerance after quantize/dequantize round-trip
  - `Requantizer` round-trip
  - Per-channel depthwise correctness
- `unit_test/embedded/` regression matrix grows: add `QUANT=1` corner. Confirms quantization path builds with `STD=0, FLOAT=0` (inference-only embedded).
- `examples/kws_cortex_m/` gets quant variant `kws_cortex_m_int8/` — same pipeline, int8 weights, CSV cycle/byte report for comparison.

## Phase 6 — Docs

- Append section to `CLAUDE.md` under Architecture Overview: "Quantization (optional, `TINYMIND_ENABLE_QUANTIZATION=1`)".
- README gets one paragraph + pointer to example.

## Risk / non-goals

- **No QAT** (quantization-aware training). Post-training quantization only — fits existing "PyTorch float training → MCU inference" workflow.
- **No int4 / mixed precision** initially. Just int8 weights/activations + int32 accumulator.
- **No fusion pass** (conv+bn+relu folded). Each layer standalone first; fusion later.
- **Per-channel only on depthwise** in v1 (TFLite minimum requirement). Other layers per-tensor.

## Order of work (suggested PRs)

1. `tinymind_platform.hpp` gate + `qaffine.hpp` types + unit test for `Requantizer`
2. `qcalibration.hpp` + tests
3. `qdense.hpp` + `qactivations.hpp` (smallest end-to-end slice)
4. `qconv2d.hpp` + `qpool2d.hpp`
5. `qdepthwiseconv2d.hpp` + `qpointwiseconv2d.hpp` (per-channel)
6. KWS int8 example + bench comparison
7. Embedded regression matrix update + docs

Each PR self-contained, existing tests stay green throughout (additive only).

## Codebase survey results (reference)

From investigation 2026-05-08:

- All composable layers parameterize on single `typename ValueType`. None separate input/weight/accum/output types.
- `cpp/neuralnet.hpp:2476` — `ValueType sum` accumulator (single type) in forward pass.
- `cpp/conv2d.hpp:113,126` — `ValueType sum = bias; sum += weight * input;` (same type for MAC and accumulation).
- `cpp/qformat.hpp:383–408` — `QValue` is pure compile-time fixed/fractional bit split. Single union storage. No runtime metadata field.
- No `scale`, `zero_point`, `requant` symbols exist outside compile-time / per-stage shifts (FFT butterfly, dropout `1/(1-p)`, gelu scale).
- `cpp/include/nnproperties.hpp:155–265` — `ValueConverter` is the only existing host→target weight conversion path. Used by `NetworkPropertiesFileManager`.
- `cpp/include/tinymind_traits.hpp` — `enable_if`, `is_floating_point` available for SFINAE at `STD=0`.

These confirm: quantization cannot retrofit cleanly into the existing single-`ValueType` templates without invasive changes. Parallel layer set is the right call.

---

# Mixed-Precision Inference Roadmap (Phases 9–16)

Continuation after Phase 8 ships. Goal: enable TinyMind to emulate common GPU-deployed models — CNN (ResNet/MobileNetV2/V3/EfficientNet), recurrent (LSTM/GRU), transformer (encoder block), and mixed-precision hybrids — across MCU through application-class CPUs (M0+, M4F/M7, Cortex-R82, Cortex-A55, x86).

Each phase = self-contained PR. Builds on prior. Ships with regression corner + at least one example. Additive only — existing Q-format and Phase 1–8 int8 layers untouched.

## Cross-Phase Invariants

- **Freestanding-clean.** Every runtime header compiles under `FLOAT=0 STD=0 QUANT=1` (M0+ deployable shape). Calibration / fold / SIMD-host helpers gated on `FLOAT && STD`.
- **Caller-owned buffers.** No malloc in runtime path. Matches existing `q*.hpp` pattern.
- **Bit-exact across SIMD gates.** Scalar reference is source of truth. SIMD specializations validated against it.
- **Embedded regression corner per phase.** `unit_test/embedded/Makefile` adds a new corner when a new gate lands.
- **Copyright.** New files Dan McLeran copyright only.

## Phase 9 — Type Bridges + Storage Tiers

**Goal:** make precisions composable. End orphaned-pipelines problem between `QValue` (Q-format) and `QAffineTensor` (int8 affine).

**Scope:**
- `cpp/qbridge.hpp` — new header:
  - `affineToQValue<QV>(int8_t q, AffineParams)` → `QValue`
  - `qValueToAffine<QV>(QValue, AffineParams)` → `int8_t`
  - `affineToFloat` / `floatToAffine` promoted from calibration helpers to runtime (freestanding-safe when `FLOAT=1`)
- New gate `TINYMIND_ENABLE_FP16` in `cpp/include/tinymind_platform.hpp`.
- `cpp/include/tinymind_fp16.hpp` — software-only `fp16_t` (IEEE 754 binary16) and `bf16_t` (bfloat16) storage structs wrapping `uint16_t`, plus scalar promote-to-float ops via `__builtin_memcpy`. No compiler-builtin `__fp16` / `_Float16` dependency — keeps the header capability-gate-clean per the Phase 14 design rule, and the storage tier compiles on any toolchain regardless of ISA. Hosts that natively support `_Float16` / `__fp16` may add a thin adapter without disturbing this header. Pure storage tier; the matching SIMD vector specialization lands in Phase 14 as `cpp/include/simd/simd_neon_fp16.hpp`.
- `unit_test/embedded/Makefile` — expand to 4-way `(FLOAT, STD, QUANT, FP16)` matrix. Add `fp16_hosted` corner.

**Tests:** `unit_test/quantization/test_qbridge.cpp` — round-trip Q8.8 ↔ int8 ↔ Q8.8 within tolerance. fp16 ↔ fp32 round-trip.

**Success criteria:** existing Q-format LSTM / attention / FFT consumable by int8 frontend output via converter call. No perf regression.

**Risk:** low. Pure new code, no edits to existing headers.

## Phase 10 — Skip-Connection + Composition Ops

**Goal:** unlock ResNet, MobileNetV2/V3, EfficientNet, U-Net architectures.

**Scope:**
- `cpp/qadd.hpp` — `QAdd<InA, InB, Out>` with two input Requantizers + one output Requantizer (TFLite ADD semantics).
- `cpp/qmul.hpp` — elementwise multiply (SE-block gating).
- `cpp/qconcat.hpp` — channel-dim concat with per-input Requantizer rescaling to common output scale.
- `cpp/qpad.hpp` — zero-pad / replicate pad (needed for SAME-padding conv).
- Promote `QConv2D` and `QPointwiseConv2D` to **per-channel** weight scales. Keep per-tensor as default template default for back-compat. Mirror depthwise pattern (`cpp/qdepthwiseconv2d.hpp:72`).

**Tests:** Boost.Test additions in `unit_test/quantization/`:
- ADD round-trip parity vs float reference
- Per-channel QConv2D parity
- ResNet-block-shaped fixture (Conv → BN-folded → ReLU → Conv → Add → ReLU)

**Example:** `examples/resnet_block_int8/` — single residual block, CSV cycle/byte report alongside `kws_cortex_m_int8`.

**Success criteria:** MobileNetV2-shaped block runs int8 end-to-end. Per-channel `QConv2D` matches TFLite reference within 1 ulp on output int8 grid.

**Risk:** medium. Per-channel touches existing layer template signature — needs careful default to avoid breaking `kws_cortex_m_int8`.

## Phase 11 — Normalization + Softmax + BN Fold

**Goal:** unlock any modern CNN trained with BatchNorm + any classifier with softmax output. Foundational for transformer (Phase 13).

**Scope:**
- `cpp/include/qcalibration.hpp` additions:
  - `foldBatchNorm(conv_w, conv_b, bn_gamma, bn_beta, bn_mean, bn_var, eps)` → fused conv weights/bias. Host-only, gated on `FLOAT && STD`.
- `cpp/qbatchnorm.hpp` — standalone int8 batchnorm for cases that cannot fold (post-pool BN). Per-channel scale + shift.
- `cpp/qlayernorm.hpp` — int8 LayerNorm: integer mean + rsqrt LUT or Newton iteration. Required for transformer.
- `cpp/qsoftmax.hpp` — int8 → int8 softmax. exp LUT + integer normalize. Output scale fixed 1/256, zero_point = -128 (TFLite convention).

**Tests:** per-layer parity vs float reference (max abs error ≤ 1 on int8 grid). Fold-pass regression: float Conv+BN vs int8 fused Conv must agree pre-quantization.

**Example:** extend `examples/kws_cortex_m_int8/` with BN-folded variant. Add `examples/mnist_int8/` with softmax output.

**Success criteria:** model with Conv+BN+ReLU+Softmax classifier deployable end-to-end int8.

**Risk:** medium. Softmax integer normalization is tricky — must avoid overflow on max-shift before exp.

## Phase 12 — Recurrent Quant (LSTM / GRU) [SHIPPED]

**Goal:** unlock sequence models — language, audio, time-series prediction.

**Scope (shipped):**
- `cpp/qlstm.hpp` — `QLSTMCell<InputStorage, WeightStorage, AccumStorage, GateActStorage, HiddenStorage, CellStorage, NumInputs, NumHidden>`. Per-gate (i, f, g, o) two-MAC pre-activation in a shared LUT input scale, sigmoid/tanh via existing `qApplyLUT`. Cell update is `f * c_prev + i * g` through two `multiplyByQuantizedMultiplier` calls; output is `o * tanh(c)` through one `Requantizer`. `CellStorage` template parameter selects `int8_t` (default, deployable) or `int16_t` (wide carry-state).
- `cpp/qgru.hpp` — `QGRUCell` with three gates (r, z, n), reset-before-multiply variant: `n_t = tanh(W_n x + R_n (r_t * h_{t-1}) + b_n)` then `h_t = (1-z_t)*n_t + z_t*h_{t-1}`. `(1-z_t)` derived in the sigmoid grid as `-z_t` (real-domain identity).
- `cpp/include/qcalibration.hpp` additions: `QLSTMScales`/`QLSTMParams`/`buildQLSTMParams` + `quantizeQLSTMBiases`, and `QGRUScales`/`QGRUParams`/`buildQGRUParams` + `quantizeQGRUBiases`. Host-only.
- New gate `TINYMIND_ENABLE_INT16_ACCUM=1` advertises the int16 cell-state corner to the embedded matrix.

**Tests (shipped):**
- `qlstm_single_step_parity_with_float_reference` — float reference vs int8 cell parity within ~5% of hidden dynamic range.
- `qlstm_int16_cell_long_sequence_stays_bounded` — 256-step drive of the int16 cell-state variant; verifies the carry-state stays well inside its range and tracks the float reference.
- `qgru_single_step_parity_with_float_reference` — float GRU reference vs int8 cell parity.

**Embedded regression:** new corner `int16_accum_freestanding` (`QUANT=1 INT16_ACCUM=1 FLOAT=0 STD=0`) exercises QLSTM (int8 + int16 cell) and QGRU at the deployable freestanding shape.

**Status:** Sinusoid example and direct float MSE bench deferred to Phase 16 (mixed-precision exemplars).

**Risk:** medium-high. Quantized recurrent state is notoriously hard; the int16 cell variant ships as an opt-in template parameter for callers running long unroll horizons.

## Phase 13 — Attention + FFT [SHIPPED]

**Goal:** unlock transformer encoders + audio front-ends.

**Scope (shipped):**
- `cpp/qattention1d.hpp` — `QAttention1D<...>`. Linear (ReLU-kernel) self-attention. Q/K/V projections inlined (sharing the MAC over X to spare one pass over the input); each MAC has its own `Requantizer`. ReLU on Q'/K' is folded by `qmin = zero_point` on the projection requantizers (matches `clampForRelu`). Caller-owned scratch buffers for Q', K', V', KV.
- `cpp/qattention_softmax.hpp` — `QAttentionSoftmax1D<...>`. Standard softmax-attention. Adds an S x S score buffer and an attention buffer (1/256 scale, zp -128 per TFLite). Score requantizer folds the `1 / sqrt(d_k)` factor via `qAttentionInvSqrt(P)`. Reuses the 256-entry int32 exp LUT from Phase 11 (`buildQSoftmaxExpLUT`).
- `cpp/qfft1d.hpp` — `QFFT1D<N>`. Radix-2 DIT FFT on int16 buffers with Q1.15 twiddle factors (caller-owned, built host-side by `buildQFFTTwiddles`). Scaled butterflies (right-shift by 1 each stage; total 1/N) keep the int16 working register bounded. `magnitudeSquared` returns int32; the int8 boundary on either side is expressed as an ordinary `Requantizer`. Inverse via the conjugate trick (unscaled).
- `cpp/qmha.hpp` — `QMultiHeadLinearAttention1D<..., NumHeads>`. Holds `NumHeads` independent `QAttention1D` heads and stacks their per-head outputs along the projection axis. Scratch buffers (Q', K', V', KV, per-head output) are reused across heads.
- `cpp/include/qcalibration.hpp` additions: `buildQFFTTwiddles` (Q1.15 sin/cos table), `QAttention1DScales` / `QAttentionSoftmaxScales` documentation structs, and `qAttentionInvSqrt(P)` for folding the score-scaling factor host-side.

**Tests (shipped):**
- `qfft_twiddle_table_unit_circle` — every twiddle satisfies cos^2 + sin^2 ~= 1 within 1 ulp at int16.
- `qfft_magnitude_spectrum_matches_float_reference` — sinusoid at bin 3 of length 16; QFFT magnitudes within 0.05 of a naive float DFT reference and the peak is at the expected bins.
- `qfft_forward_inverse_round_trip_is_close_to_identity` — int16 forward/inverse recovers the input within a few hundred LSBs of int16 noise.
- `qattention_linear_parity_with_float_reference` — small S/E/P shape; output within 8% of the float linear-attention reference's dynamic range.
- `qattention_softmax_parity_with_float_reference` — same shape with TFLite-conventional softmax; output within 15% of the float softmax-attention reference.
- `qmha_stacks_two_identical_heads` — two identical heads emit byte-identical stacked output to the single-head ground truth.

**Embedded regression:** `quant_freestanding` corner now exercises `QFFT1D`, `QAttention1D`, `QAttentionSoftmax1D`, and `QMultiHeadLinearAttention1D` so the new headers stay free of `<cmath>` / `<type_traits>` / stdlib.

**Example:** `examples/transformer_encoder_int8/` — single encoder block (LayerNorm + linear attention + Add + LayerNorm + Dense+ReLU+Dense + Add). Calibrates every intermediate tensor on a small synthetic dataset, quantizes weights and biases, runs end-to-end on int8 and reports max-abs error vs the float reference (~2 % of output range on the bundled dataset; tolerance set at 40 %).

**Success criteria:** transformer encoder block runs int8 end-to-end. ✓ shipped. QFFT magnitude spectrum tracks float DFT within ~0.5 dB on the bundled sinusoid. ✓ shipped.

**Risk:** high. Attention numerics sensitive — linear-attention variant first reduced risk; softmax variant ships alongside.

## Phase 14 — SIMD Performance Backend [SHIPPED]

**Goal:** close the cycle-count gap with TFLite-Micro / XNNPACK on any target whose silicon ships the relevant SIMD ISA extension. Scalar path untouched and remains the default everywhere.

### Design rule — capability gates, not CPU gates

SIMD presence is orthogonal to CPU complex. Verified against Arm documentation (developer.arm.com, GCC AArch64 Options, Arm Neon intrinsics reference):

- **Cortex-A55** ships with NEON, FPU, and Crypto as optional RTL components — Arm publishes >3000 distinct A55 RTL configurations. NEON is not guaranteed.
- **Cortex-R82** "optionally includes the latest Neon instructions to greatly accelerate machine learning workloads with capabilities such as Dot Product support." NEON is opt-in at synthesis time.
- **AArch64** itself permits `+nosimd` / `-mgeneral-regs-only` builds (GCC AArch64 docs). Bare-metal and kernel-mode code regularly runs without NEON.
- **Helium (MVE)** is optional on Armv8.1-M cores (M55 / M85 / M52); MVE-I (integer) and MVE-F (float) are independently selectable — a core can implement MVE-I without MVE-F.

Naming a gate after a CPU therefore presumes capability the silicon may not have. Library headers must key on ISA capability, never on CPU model.

### Design rule — gates honor Arm's architectural prerequisite chain

Gates are CPU-agnostic but **not fully independent of each other.** Per Arm intrinsics reference and GCC AArch64 docs, the architecture defines a strict prerequisite chain:

| Capability | Architectural prerequisite |
|------------|----------------------------|
| `SIMD_NEON` | (implicitly `FP`) |
| `SIMD_NEON_DOTPROD` (SDOT/UDOT) | `SIMD_NEON` — dot product instructions live inside the Advanced SIMD instruction set; there is no "dotprod without NEON" |
| `SIMD_NEON_FP16` (FEAT_FP16 vector forms) | `SIMD_NEON` + `FP16` scalar extension |
| `SIMD_SVE` | `SIMD_NEON` (GCC: "+sve also enables Advanced SIMD and floating-point instructions") |
| `SIMD_SVE2` | `SIMD_SVE` |
| `SIMD_BF16`, `SIMD_I8MM` | `SIMD_NEON` (or `SIMD_SVE`) |
| `SIMD_HELIUM_MVE_I` | M-profile only — mutually exclusive with `SIMD_NEON` / `SIMD_SVE` |
| `SIMD_HELIUM_MVE_F` | M-profile only — independent of `SIMD_HELIUM_MVE_I` per Arm Helium docs |
| `SIMD_AVX_VNNI` (`VPDPBUSD`) | `SIMD_AVX2` (Alder Lake+ ships VNNI without AVX-512) |
| `SIMD_AVX512F` | (x86 baseline AVX assumed) |
| `SIMD_AVX512_VNNI` | `SIMD_AVX512F` |

Library policy: each `simd_*.hpp` header opens with a `static_assert` enforcing its row of this table. Caller cannot enable a dependent gate without its prerequisites; misconfiguration fails at compile time with a readable message.

### Scope

Specialize inner loops behind capability-named feature gates. Library otherwise unchanged.

- `cpp/include/simd/` — new directory. One header per capability, each declaring its prerequisites via `static_assert`:
  - `simd_neon.hpp` — Armv7 / Armv8-A Advanced SIMD baseline (`VMULL`, `VPADDL`). Gate `TINYMIND_ENABLE_SIMD_NEON=1`.
  - `simd_neon_dotprod.hpp` — `SDOT`/`UDOT` Armv8.2-A FEAT_DotProd. Gate `TINYMIND_ENABLE_SIMD_NEON_DOTPROD=1`. Requires `SIMD_NEON`.
  - `simd_neon_fp16.hpp` — Armv8.2-A FEAT_FP16 vector forms (`float16x8_t` arithmetic). Gate `TINYMIND_ENABLE_SIMD_NEON_FP16=1`. Requires `SIMD_NEON`. Pairs with Phase 9 `fp16_t` storage.
  - `simd_sve.hpp` — Scalable Vector Extension, vector-length-agnostic loops. Gate `TINYMIND_ENABLE_SIMD_SVE=1`. Requires `SIMD_NEON`.
  - `simd_sve2.hpp` — SVE2 superset. Gate `TINYMIND_ENABLE_SIMD_SVE2=1`. Requires `SIMD_SVE`.
  - `simd_helium_mve_i.hpp` — Armv8.1-M MVE-I (integer). Gate `TINYMIND_ENABLE_SIMD_HELIUM_MVE_I=1`. Mutually exclusive with NEON / SVE (M-profile only).
  - `simd_helium_mve_f.hpp` — Armv8.1-M MVE-F (float). Gate `TINYMIND_ENABLE_SIMD_HELIUM_MVE_F=1`. Independent of MVE-I (a core can implement either alone).
  - `simd_avx2.hpp` — x86 `PMADDUBSW` fallback path. Gate `TINYMIND_ENABLE_SIMD_AVX2=1`.
  - `simd_avx_vnni.hpp` — `VPDPBUSD` (Alder Lake+ / Sapphire Rapids on the AVX-VNNI side). Gate `TINYMIND_ENABLE_SIMD_AVX_VNNI=1`. Requires `SIMD_AVX2`.
  - `simd_avx512f.hpp` — AVX-512 foundation. Gate `TINYMIND_ENABLE_SIMD_AVX512F=1`.
  - `simd_avx512_vnni.hpp` — AVX-512-VNNI. Gate `TINYMIND_ENABLE_SIMD_AVX512_VNNI=1`. Requires `SIMD_AVX512F`.
- `cpp/include/threading.hpp` — optional `#pragma omp parallel for` over output-channel dim for conv layers. Gate `TINYMIND_ENABLE_OPENMP=1`. Orthogonal to every SIMD gate.

### Gate semantics

- All `TINYMIND_ENABLE_SIMD_*` default to `0`. Scalar fallback is the default body of every layer.
- No `#ifdef __ARM_NEON` / `#ifdef __AVX2__` auto-detection in library headers. Build system (Makefile / CMake) translates compiler flags into matching `TINYMIND_ENABLE_SIMD_*=1` defines. Keeps library decoupled from toolchain auto-detection quirks.
- Gates are CPU-agnostic (do not assume A55 → NEON, R82 → NEON-DotProd, M55 → Helium, etc.) but honor Arm's architectural prerequisite chain (see table above). Misconfiguration like `DOTPROD=1, NEON=0` is rejected at compile time by `static_assert`.
- No runtime CPU dispatch (`cpuid` / `getauxval` / `__builtin_cpu_supports`). Library compiles for one ISA target per build. Fat-binary / multi-arch dispatch is caller's problem.

### Tests

SIMD specializations must be **bit-exact** with the scalar reference. Boost.Test corner per gate combo in `unit_test/quantization/`. `unit_test/embedded/Makefile` adds:

- `simd_disabled` corner — every `SIMD_*=0`, proves scalar path still builds and runs at the deployable freestanding shape.
- One corner per major gate (`simd_neon_dotprod_hosted`, `simd_avx_vnni_hosted`, `simd_helium_mve_i_freestanding`, …) on hosts where the cross-compiler is available; CI matrix flags absent toolchains as "skipped" rather than "failed".
- A compile-failure regression test that asserts `DOTPROD=1, NEON=0` fails with the static_assert message (proves the prerequisite chain is enforced, not documented-only).

### Example

`examples/perf_matrix/` — runs a MobileNetV2 block across every available SIMD gate combo, emits CSV with cycle counts plus the set of `SIMD_*` flags active per row so deltas are interpretable. `examples/kws_cortex_m_int8/` and `examples/transformer_encoder_int8/` bench harnesses also extend their CSV header with the active `SIMD_*` flag set.

### Success criteria

- ≥6× speedup on Arm with `SIMD_NEON_DOTPROD=1` (whatever the CPU complex — A55, R82, Neoverse-class) vs scalar for conv-dominated workload.
- ≥10× speedup on x86 with `SIMD_AVX_VNNI=1` vs scalar.
- Scalar fallback unchanged: `simd_disabled` corner of `unit_test/embedded` continues to pass byte-identically to pre-Phase-14 outputs.
- Bit-exact output across every gate combo on the same model.

### Risk

Medium. Bit-exactness across SIMD widths needs careful order-of-accumulation handling (`SDOT` reduces 4 lanes per accumulator slot; AVX-VNNI reduces 4 int8 per int32 slot — accumulation order is fixed per ISA and the scalar reference must match by construction, not by chance). Build-system intrinsics matrix is friction but capability-gate design keeps the library side clean.

### Status (shipped)

- 12 gates landed in `cpp/include/tinymind_platform.hpp` (11 SIMD + OpenMP), all defaulting to 0. Scalar path is byte-identical to the pre-Phase-14 build.
- 12 backend headers under `cpp/include/simd/`: NEON baseline, NEON DOTPROD (SDOT/UDOT), NEON FP16 vector, SVE (svdot_s32), SVE2 (forward-compatibility), Helium MVE-I (vmladavaq_s8), Helium MVE-F, AVX2 (cvtepi8_epi16 + madd_epi16, deliberately avoiding PMADDUBSW for bit-exactness), AVX-VNNI (VPDPBUSD via uint8-shift trick), AVX-512F, AVX-512-VNNI. Each header opens with a static_assert enforcing Arm's prerequisite chain.
- `cpp/include/simd/simd_dispatch.hpp` provides the public `int8DotWithZeroPoint` and templated `dotProductWithZeroPoint<Input, Weight, Accum>`. `QDense`, `QConv2D`, and `QConv2DPerChannel` route their inner reduction through it. Float / int16 / non-int8 type combos transparently fall back to the scalar template body.
- `cpp/include/threading.hpp` provides `TINYMIND_PARALLEL_FOR_OUTER`. The output-filter loop of `QConv2D` and `QConv2DPerChannel` is annotated.
- `unit_test/embedded/Makefile` gains the `simd_disabled` corner (every SIMD_*=0 at QUANT=1 FLOAT=0 STD=0) and the `simd_prereq_regressions` target that locks `AVX_VNNI=1, AVX2=0` and `AVX512_VNNI=1, AVX512F=0` as compile failures.
- `unit_test/quantization/` adds five Phase 14 tests: dispatch bit-exactness across pathological lengths (0, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257, 1024 with `zp` sweep), INT8 extreme-value patterns, full-layer `QDense` parity, full-layer `QConv2D` parity, and the `activeBackendName` reporter. All 109 tests pass under scalar, AVX2, AVX-512F, and AVX-512-VNNI builds on the reference Tiger Lake host.
- `examples/perf_matrix/` builds the same int8 conv + dense block under scalar / AVX2 / AVX-512F / AVX-512-VNNI and emits one CSV row per binary with `active_backend, conv/dense per-call us, output_checksum`. Reference run: scalar 305 us/call conv → AVX2 237 us/call (~28% faster); all backends agree on the output checksum byte-for-byte, locking the bit-exact invariant.

## Phase 15 — Calibration Upgrades + Importer Tooling [SHIPPED]

**Goal:** reduce accuracy gap vs PyTorch / TFLite reference. Lower friction to deploy.

**Scope (shipped):**
- `cpp/include/qcalibration.hpp` additions:
  - `PercentileObserver` — record samples, query `[lower_pct, upper_pct]` range. Clips outliers ahead of `computeAffineParamsAsymmetric`.
  - `KLDivergenceObserver` — TensorRT-style entropy calibration. Two-pass over a 2048-bin histogram, sweeps candidate clip thresholds, returns the one that minimizes KL between the reference distribution and the int8-quantized distribution. Output is an absmax for symmetric quantization.
  - `crossLayerEqualizeDense` / `crossLayerEqualizeConv2D` — Nagel-paper Cross-Layer Equalization. Per-channel `s = sqrt(r1 / r2)` rebalances the upstream output range against the downstream input range. ReLU / identity activations make the transform output-preserving in the float domain (positively homogeneous).
- `apps/import_pytorch/tinymind_import.py` — Python importer module. Caller assembles `Dense` / `Conv2D` / `BatchNorm2D` / `ReLU` / `Sigmoid` / `Tanh` / `Softmax` descriptors carrying numpy weights from `torch.state_dict`. `fuse_layers` detects Conv2D-then-BatchNorm and folds via the same math as `foldBatchNorm`. `calibrate` streams the calibration dataset through the float reference with `MinMaxObserver` / `PercentileObserver` / `KLDivergenceObserver` per layer. `quantize_weights` emits symmetric int8 weights + int32 biases. `emit_weights_header` writes a TinyMind-format `weights.hpp`. Top-level `import_pytorch_model` wraps the four passes.
- `apps/import_onnx/tinymind_import_onnx.py` — QDQ-format ONNX importer. Walks `QuantizeLinear` / `DequantizeLinear` / `QLinearConv` / `QLinearMatMul` / `Relu` / `Sigmoid` / `Tanh` / `Softmax` nodes, extracts the per-tensor `(scale, zero_point)` and int8 weight / int32 bias initializers, emits the same TinyMind-format `weights.hpp`. The `onnx` Python package is imported lazily so the emitter half is usable without it.

**Tests (shipped):**
- `percentile_observer_clips_outliers` / `percentile_observer_empty_returns_zero_range` — heavy-tail vs uniform data, empty buffer edge case.
- `kl_divergence_observer_finds_clip_threshold` / `kl_divergence_observer_empty_returns_zero` — Gaussian + outliers; threshold lands below absmax, well below the outlier mass. Empty edge case returns 0.
- `cross_layer_equalize_dense_preserves_relu_output` / `cross_layer_equalize_dense_skips_zero_channels` — 100:1 channel imbalance forced; output unchanged after CLE; zero-row channel is left alone.
- `cross_layer_equalize_conv2d_preserves_relu_output` — same invariant for the Conv2D variant with OHWI weight layout.

**Example:** `examples/import_demo/` — end-to-end Phase 15 demo. The C++ binary (`import_demo.cpp`) carries a deterministic 3-8-4-2 MLP, drives a 64-sample synthetic calibration set through `RangeObserver` / `PercentileObserver` / `KLDivergenceObserver` plus `crossLayerEqualizeDense`, then runs both float and int8 forward and prints the max-abs error (~0.004 on the bundled seed; tolerance 0.08). Standalone — no torch dependency. `demo.py` is the production-flow counterpart that consumes `torch.state_dict` and drives `apps/import_pytorch/tinymind_import` to emit a real `weights.hpp`.

**Success criteria:** one-command import from PyTorch quantized model to deployable C++ stack. ✓ shipped via `apps/import_pytorch/tinymind_import.import_pytorch_model`.

**Risk:** low-medium. Mostly Python tooling, isolated from runtime.

## Phase 16 — Mixed-Precision Exemplars + Verification [SHIPPED]

**Goal:** prove end-to-end models really run. Lock with regression tests.

**Scope (shipped):** four reference exemplars, one directory per model:

1. `examples/resnet18_block_int8/` — int8 ResNet-18 stem + one basic-block stage (`QPad2D` → `QConv2DPerChannel 7x7 s=2` → `qrelu` → `QMaxPool2D` → basic block: `QPad2D` → 3x3 conv → `qrelu` → `QPad2D` → 3x3 conv → `QAdd` skip → `qrelu` → `QGlobalAvgPool2D` → `QDense`). Exercises Phase 10 `QPad2D` / `QConv2DPerChannel` / `QAdd` at deeper spatial dimensions than the original `resnet_block_int8`. Demonstrates that `QMaxPool2D`, `qreluBuffer`, and `QGlobalAvgPool2D` are pass-throughs on the int8 affine grid (max, clamp, integer-mean all preserve `(scale, zero_point)`), so consecutive layers reuse the upstream grid rather than burning new requantizers.
2. `examples/mobilenetv2_int8/` — int8 MobileNetV2-shaped pipeline. Stride-2 stem + one stride-1 inverted-residual block with skip + one stride-2 inverted-residual block without skip, then GAP + dense. Linear bottlenecks per MNv2 convention (no `qrelu` after the 1x1 projection). Exercises the `expand → DW → project` triple — the load-bearing primitive of MNv2 / V3 / EfficientNet.
3. `examples/transformer_encoder_int8/` — already present from Phase 13; Phase 16 wires it into the integration suite with a matching `--golden` mode.
4. `examples/mixed_precision_kws/` — mixed-precision exemplar. int8 `QDense` frontend → Phase 9 `affineI8ToFp16` bridge → fp16 linear-attention head with residual skip + mean-pool → Phase 9 `fp16ToAffineI8` bridge → int8 `QDense` classifier. Requires `TINYMIND_ENABLE_FP16=1`. Inner attention arithmetic runs in float promoted from `fp16_t`; on targets that ship vector fp16 arithmetic the promote pair is near-free, on every other target it is the cost of admission for fp16 storage on an MCU.

Each exemplar Makefile exposes three modes: `make run` (parity report vs float reference; PASS within 40-50 % of output dynamic range), `make bench` (CSV cycle/byte report — one row per layer, mirrors `examples/kws_cortex_m_int8/`), `make golden` (int8 byte stream for the bundled deterministic test set in a stable text format). Each ships with a per-precision-tier README.

**Tests (shipped):** `unit_test/integration/` — new Boost.Test suite. One fixture per exemplar shells out to the example binary with `--golden` via `popen()` and asserts the emitted byte stream matches a baked-in expected string. The exemplar binaries are deterministic (hand-crafted weights, fixed synthetic dataset, pure-integer forward), so the output is invariant across SIMD gate combos by Phase 14's bit-exactness guarantee. Any silent drift in the example pipeline, the `qaffine.hpp` requantizer, the `qcalibration.hpp` helpers, or any SIMD specialization that claims bit-exactness trips the test.

**Success criteria:** repo ships four working mixed-precision exemplars with reproducible benchmark numbers, locked at byte granularity by the integration suite. ✓ shipped.

**Risk:** low. Final phase — every runtime component landed in Phases 9-14, every host-side helper in Phase 15.

## Dependency Graph

```
Phase 9 (bridges, fp16)
   |
   v
Phase 10 (add/concat/per-channel)
   |
   v
Phase 11 (BN/LN/softmax/fold)
   |
   +--> Phase 12 (LSTM/GRU)        --+
   |                                 |
   +--> Phase 13 (attention/FFT)   --+--> Phase 16 (exemplars)
                                     |
Phase 14 (SIMD) -- independent ------+
Phase 15 (importer) -- needs Phase 11 fold
```

**Parallelizable:** Phase 14 can ship anytime after Phase 10. Phases 12 and 13 are independent of each other. Phase 15 needs Phase 11.

## Effort / Risk Summary

| Phase | Size | Risk | Unlocks |
|-------|------|------|---------|
| 9 | S | low | Q-format ↔ int8 interop, fp16 storage |
| 10 | M | medium | ResNet / MobileNetV2/V3 / EfficientNet |
| 11 | M | medium | Conv+BN+ReLU+Softmax pipelines |
| 12 | L | high | LSTM / GRU sequence models |
| 13 | L | high | Transformer encoder, audio FFT |
| 14 | L | medium | R82 / A55 / x86 perf parity |
| 15 | M | low-medium | One-command PyTorch / ONNX import |
| 16 | M | low | Regression-locked mixed-precision exemplars |

Total: roughly 8–12 PRs depending on splitting. Phase 9 alone unblocks usefulness on bigger CPUs. Phases 10+11 alone close most of the CNN-class gap. Phases 12+13 are the transformer / sequence-model unlocks. Phase 14 is the perf unlock for application-class CPUs.

## Target-Tier Implications

CPU complex and SIMD capability are orthogonal — the rows below describe *typical* configurations, not guarantees. Arm publishes thousands of distinct RTL configurations per core (Cortex-A55 alone has >3000), and NEON / Crypto / FPU are often optional components. The library never assumes a capability from a CPU name; caller sets the matching `TINYMIND_ENABLE_SIMD_*` gates per their actual silicon, and the headers' `static_assert`s enforce Arm's architectural prerequisite chain.

- **M0+ / freestanding:** Phases 9–13 land new ops but every runtime header stays freestanding-clean. All Phase 14 `SIMD_*` gates default off; scalar fallback is the path. Phase 15 tooling host-only.
- **M4F / M7:** scalar by default; optional Phase 9 fp16 storage. No SIMD gate applies (Helium MVE is Armv8.1-M+).
- **M55 / M85 / M52 (Armv8.1-M):** opt-in `TINYMIND_ENABLE_SIMD_HELIUM_MVE_I=1` and/or `TINYMIND_ENABLE_SIMD_HELIUM_MVE_F=1` *if* the part is built with the matching MVE flavor. Arm permits MVE-I without MVE-F; both gates are independently selectable. M55 / M85 / M52 configured without MVE fall back to the M4F/M7 story.
- **Cortex-R82 (Armv8-R 64-bit):** opt-in `TINYMIND_ENABLE_SIMD_NEON` and `TINYMIND_ENABLE_SIMD_NEON_DOTPROD` *if* the part was built with NEON. Per Arm, R82's NEON unit is optional at synthesis time. R82 configured without NEON runs the scalar path; real-time determinism story preserved either way.
- **Cortex-A55 and other Armv8.2-A application cores:** Arm's A55 product page lists NEON, Crypto, and the FPU as *optional* components in the RTL. The common high-throughput config is `TINYMIND_ENABLE_SIMD_NEON=1` + `TINYMIND_ENABLE_SIMD_NEON_DOTPROD=1` + optional `TINYMIND_ENABLE_SIMD_NEON_FP16=1` + optional `TINYMIND_ENABLE_OPENMP=1`. A55 configurations that omit NEON exist; caller sets gates accordingly.
- **Cortex-A510 / Cortex-A715 / Neoverse V1 / Neoverse V2 / Neoverse N2:** add `TINYMIND_ENABLE_SIMD_SVE=1` (or `SVE2=1`) for vector-length-agnostic loops, on top of NEON / dotprod where present. Neoverse V2 / N2 ship SVE2; V1 ships SVE(1).
- **x86:** `TINYMIND_ENABLE_SIMD_AVX2=1` is the floor on anything modern; `TINYMIND_ENABLE_SIMD_AVX_VNNI=1` is the int8 headline on Alder Lake+ (and on Sapphire Rapids' AVX-VNNI path). `TINYMIND_ENABLE_SIMD_AVX512F=1` plus `TINYMIND_ENABLE_SIMD_AVX512_VNNI=1` on Sapphire Rapids+ also unlocks Phase 9 bf16 storage paths.

## Non-Goals (Still)

- **No QAT** in this roadmap. Post-training quantization remains the deployment path.
- **No sub-4-bit / mixed precision below int8.** Storage tier list (in mixed-precision peer relationship via Phase 9 bridges and Phase 17 pure-integer bridges) is {int8 affine, int16-accum, fp16, bf16, fp32, **Q-format `QValue<I,F>`** as a first-class peer — Phase 17 closes the loop with integer-only `affineToQValueInt` / `qValueToAffineInt` so the Q-format tier participates in hybrid models at the deployable freestanding shape `FLOAT=0 STD=0 QUANT=1`}.
- **No dynamic / runtime model loading.** Compile-time template shapes remain — codegen-from-PyTorch flow is the integration model (Phase 15).

## Phase 17 — Pure-Integer Q-format <-> int8 Bridge + Hybrid Importer [SHIPPED]

**Goal:** close the gap for the offline-training -> embedded-inference story where a model wants the int8 affine grid at the boundaries (PyTorch / TF / ONNX QDQ export shape) but a Q-format middle tier (existing `NeuralNet<Q8.8>` MCU code, or a hidden layer that prefers `QValue`'s compile-time fixed/fractional bit split). Phase 9 added the float-mediated `qValueToAffine` / `affineToQValue` bridges; Phase 17 ships the pure-integer counterparts so the inference path needs no `<cmath>` and no float at runtime.

**Scope (shipped):**
- `cpp/qbridge.hpp` additions: `AffineToQValueIntParams<QV>` / `QValueToAffineIntParams<QV>` (integer triples) plus `affineToQValueInt` / `qValueToAffineInt` (+ buffer variants). Uses the same Q0.31 `multiplyByQuantizedMultiplier` primitive that `Requantizer` does — no new runtime dependency. Gated on `TINYMIND_ENABLE_QUANTIZATION`, independent of `TINYMIND_ENABLE_FLOAT`. Host-side helper builders `buildAffineToQValueIntParams<QV>` / `buildQValueToAffineIntParams<QV>` gated on `FLOAT && STD`.
- `apps/import_pytorch/tinymind_import.py` additions: `QFormatDense` layer descriptor (Q-format dense weights/biases emitted as raw QValue integers, no scale or zero_point at runtime), `HybridBoundary` precision-tier transition descriptor, and `quantize_multiplier` / `quantize_qformat_weights` helpers. The emitter writes precomputed `(multiplier, shift, zero_point)` triples (plus `qmin`/`qmax` on the `qvalue_to_affine` side) directly into `weights.hpp` so the deployable target consumes them as data.
- `apps/import_onnx/README.md` + `apps/import_pytorch/README.md` document the TensorFlow / Keras path via `tf2onnx` + `onnxruntime.quantization.quantize_static(quant_format=QuantFormat.QDQ)` plus the hybrid `QFormatDense` + `HybridBoundary` flow.

**Tests (shipped):**
- `qbridge_int_affine_to_qvalue_matches_float_bridge` / `qbridge_int_qvalue_to_affine_matches_float_bridge` — pure-integer bridge stays within 1 LSB of the float bridge across the int8 / Q88 grid.
- `qbridge_int_round_trip_within_tolerance` — float -> Q88 -> int8 (integer bridge) -> Q88 (integer bridge) -> float closes back within one affine LSB plus one QValue LSB.
- `qbridge_int_qvalue_to_affine_saturates` — out-of-range Q88 inputs saturate to `[qmin, qmax]`.
- `qbridge_int_buffer_round_trip` — buffer-variant parity.
- `unit_test/embedded/embedded_smoke_test.cpp` exercises `affineToQValueInt` / `qValueToAffineInt` in the `quant_freestanding` corner, confirming the integer bridge stays freestanding-clean (no `<cmath>`, no `<type_traits>`, no stdlib).

**Example:** `examples/mixed_precision_mlp_int8_qformat/` — int8 `QDense` -> `qrelu` -> Phase 17 `affineToQValueInt` bridge -> Q8.8 dense matvec -> Phase 17 `qValueToAffineInt` bridge -> int8 `QDense` classifier. `make run` reports max-abs error vs the float reference (~0.005 on the bundled dataset, well below the 60 %-of-output-range tolerance). `make golden` emits a deterministic int8 byte stream that the new `mixed_precision_mlp_int8_qformat_golden_match` integration fixture in `unit_test/integration/` locks at byte granularity.

**Success criteria:** an offline-trained model with one or more Q-format hidden layers and int8 affine boundaries deployable end-to-end at `FLOAT=0 STD=0 QUANT=1`. ✓ shipped.

**Risk:** low. Pure addition — no edits to existing runtime headers' behavior at any pre-Phase-17 gate combination.

