# AGENTS.md

Guidance for coding agents working in this repository. `CLAUDE.md` imports this
file, so there is one source of truth; edit this file, not the shim.

tinymind is a **header-only C++ template library** for neural networks and
Q-learning on embedded systems. The project name is always lowercase
(`tinymind`) in prose, docs, charts and release notes; identifiers keep their
own spelling.

## Where the detail lives

This file holds the rules that apply everywhere. Each area below has its own
`AGENTS.md`; read it before changing anything in that directory.

| Working on | Read |
|---|---|
| Any header under `cpp/`: header rules, feature gates, library map | `cpp/AGENTS.md` |
| Embedded gate matrix, Arm cross-compiles, QEMU, heap-free, MSan | `unit_test/embedded/AGENTS.md` |
| Which test suite covers what | `unit_test/AGENTS.md` |
| CBMC proofs | `unit_test/formal/AGENTS.md` |
| SIMD backends, dispatch order, Arm SIMD CI | `cpp/include/simd/AGENTS.md` |
| Fuzz harnesses and corpus | `fuzz/AGENTS.md` |
| Examples, plots, re-measuring results | `examples/AGENTS.md` |
| Importers, LUT generator, ANFIS trainer | `apps/AGENTS.md` |
| On-target benchmarks, bench harness | `benchmarks/AGENTS.md` |
| Any optional tier (`cpp/q*.hpp`, LTC/CfC, attention, trees, bridges) | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |

## Build & Test Commands

```bash
# Build and run all tests and examples
make check

# Build and run one test suite (every directory under unit_test/ works the same way)
cd unit_test/nn && make clean && make && make run
cd unit_test/qformat && make clean && make && make run
cd unit_test/kan && make clean && make && make run

# Run a single test binary directly
cd unit_test/nn/output && ./nn_unit_test

# Build examples individually (make release for -O3)
cd examples/xor && make clean && make
cd examples/dqn_maze && make clean && make
```

Compiler flags: `-Wall -Wextra -Werror -Wpedantic` (debug builds also add `-ggdb`; release adds `-O3`).

The `BOOST_HOME` environment variable must be set to the Boost installation path (used by unit test Makefiles).

## Hard rules

### Language floor

**Language floor is C++14** for everything in `cpp/` (and the tests/examples), enforced by `make check-cxx14` (CI job "C++14 language floor"). Most Makefiles pass no `-std`, so day-to-day builds run at the compiler default (C++17) and will not notice a C++17-only construct; `check-cxx14` is what does. No `if constexpr`, inline variables, fold expressions, structured bindings, `std::optional`/`std::variant`/`std::filesystem`, etc. in `cpp/`. A `static constexpr` data member that is read at run time (indexed, bound to a reference, address taken) needs an out-of-class definition under `#if TINYMIND_CPLUSPLUS < 201703L` (see `fft1d.hpp`); only link-time builds catch its absence, which is why the gate runs the suites and not just a syntax pass. `TINYMIND_CPLUSPLUS` (from `tinymind_platform.hpp`) reads `_MSVC_LANG` on MSVC, where `__cplusplus` lies. Public `static constexpr` constants are documented (README, Requirements) as use-by-value under C++14 rather than each carrying a definition; tests that use one by reference (`BOOST_TEST(T::N == 3u)` binds by reference) must copy it to a local or the type must define it.

### Library code

The rules for headers under `cpp/` (include `tinymind_platform.hpp` first, feature gates default to 0, `#pragma once`, no dynamic allocation, policy-based design) and the map of the library are in `cpp/AGENTS.md`.

### Fixed-point training arithmetic

- **Never write a raw `learningRate * gradient` in an SGD update.** The gradient step uses `multiplyRoundNearest` and the momentum and acceleration terms use `multiplyRoundTowardZero`, both from `cpp/include/tinymind_rounding.hpp` (the QValue overloads are in `qformat.hpp`, found by argument-dependent lookup). The default `TruncatePolicy` floors, so a raw product turns a sub-LSB step into 0 when positive and -1 LSB when negative, and every weight drifts down; in Q8.8 that stops training (#267). Rounding the momentum term to nearest instead lets a one-LSB delta reproduce itself forever (#266). A new trainable layer needs a case in `unit_test/nn/sgd_step_rounding_tests.hpp`; the rules are documented in `docs/neural-networks.md`.
- **A change to training arithmetic (rounding, optimizers, update rules) is judged on every example that trains, over 10 seeds, master against branch** -- not on the example that motivated it and not on the default seed. Protocol: weight seed 1, data seeds 1 to 10, compare mean and standard deviation, and report the default-seed number as well. In #267 a first fix passed every unit test and the motivating example but broke `har_activity` and the Q16.16 LSTMs; an example whose learning rate was tuned against a bug may need retuning. See `examples/AGENTS.md`.

### Gates and checks

- **A new sanitizer or CI gate must be shown to fail before it is trusted** — plant the defect it exists to catch and confirm it is reported. `check-msan` is why this rule is written down rather than assumed: it runs at `-O0` because at `-O1` a planted uninitialized read is optimized away and the job passes while detecting nothing, so raising the level to tidy up build flags would leave the gate green and blind.
- Every build that is meant to fail to compile carries `$(EXPECT_FAIL)` (`-DTINYMIND_EXPECT_COMPILE_FAILURE=1`); `tools/filter_compile_commands.py` drops those from `compile_commands.json` before `make tidy`, so a new negative check must use it too or clang-tidy reports its planted failure as a library error.
- A green check or a zero exit status is not evidence that something happened: confirm the result directly (the log names the backend, the binary contains the instruction, the number moved).

### Style

- Allman braces: opening brace on its own line, `else` on its own line. Lambdas keep the trailing brace.
- Preserve each file's line endings. Some sources are CRLF (`cpp/neuralnet.hpp`, `cpp/activationFunctions.hpp`, `cpp/fixedPointTransferFunctions.hpp` among them) and the rest are LF; a scripted edit must not convert a file.
- New files carry the MIT header with `Copyright (c) <year> Dan McLeran` and no other copyright holder (see `cpp/include/tinymind_rounding.hpp`).
- Public-facing text (release notes, commit messages, PR bodies, docs) is full prose.

### Workflow

- `master` is PR-gated: branch, push, open a PR, wait for the required checks, squash-merge, delete the branch. Never push to `master` directly.
- Branch protection names the required checks by their literal CI job name, so renaming a required job blocks every merge until the protection rule is updated.
- Python tooling (matplotlib, numpy, importers, Bitwuzla) goes in a venv, never the system Python and never `pip install --break-system-packages`. Plots use the gitignored `.venv-plot/` at the repo root.

## Before pushing

| Change touches | Run |
|---|---|
| Anything | `make check` |
| `cpp/` | `make check-cxx14`, `make check-clang`, `make header-selfcheck` |
| `cpp/` or `fuzz/` | `make check-ci-image` |
| `fuzz/` | `make -C fuzz agree` and `make -C fuzz check-coverage` |
| Embedded corners, SIMD, sanitizers | the targets named in that directory's `AGENTS.md` |
| An example's output | re-run it, re-measure every number quoted in its README and `docs/examples/` page, regenerate its plot |

Toolchain gates at the top level:

- Two more g++-only checks hang off the top-level `make analyze`: `header-selfcheck` (HARD — each `cpp/**/*.hpp` compiled standalone behind a `tinymind_platform.hpp` prelude; a header that only builds via bundled include order fails, so every header carries its own `<cstddef>`/`<cstdint>` includes) and `warnings-strict` (advisory — `-Wconversion`/`-Wsign-conversion`/`-Wdouble-promotion` over the fixed-point paths).
- Two top-level clang targets cover second-toolchain conformance: `check-clang` (HARD — the embedded gate matrix plus the `CLANG_SUITES` Boost suites built and run with `clang++`) and `check-clang-avx2` (HARD — the `TSAN_SUITES` dispatch consumers with the AVX2 backend compiled in).
- `CLANG_SUITES` omits only `unit_test/integration` (it shells out to already-built example binaries, so it exits 127 on a clean tree); `check-clang-integration` (HARD) covers it instead by building the twelve exemplars it consumes with `clang++` first, which also proves the int8 pipelines are bit-exact across compilers since the goldens are shared.
- `check-clang-examples` (HARD) builds every example under `clang++` — the list is discovered from the tree rather than enumerated, so a new example is covered when it is added; it is a regression guard, since the two examples that were failing the clang build (`maze`, `dqn_maze`) were fixed in #174.
- CI runs on `ubuntu-latest`, whose compilers can be newer than the host's (Ubuntu 26.04: GCC 15 at `-march=x86-64-v3` via the amd64v3 archive variant, cppcheck 2.19, clang 21); `make check-ci-image` (podman, `tools/ci-image.sh`) runs the main gates in a container with the runner's packages on a copy of the working tree — run it before pushing a change that touches `cpp/` or `fuzz/`, and when GitHub moves `ubuntu-latest`, change `CI_IMAGE_BASE` in `tools/ci-image.sh`.
- Sub-Makefiles assign `CC=g++`, but a command-line override beats a makefile assignment and propagates through `MAKEFLAGS`, so `make CC=clang++ ...` needs no sub-Makefile edits.
- `check-msan` (HARD) runs the embedded gate corners and the fuzz corpus under MemorySanitizer at `-O0`; see `unit_test/embedded/AGENTS.md` before touching its flags.

## Tests, Examples & Apps

Full detail in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Per-directory rules:

- **`unit_test/`** — one Boost.Test suite per directory; map in `unit_test/AGENTS.md`.
- **`examples/`** — every runnable example writes a CSV and ships a `plot.py`; conventions in `examples/AGENTS.md`.
- **`apps/`** — host-side generators, importers and trainers; `apps/AGENTS.md`.
- **`benchmarks/`** — Arm-only on-target comparisons, not built in CI; `benchmarks/AGENTS.md`.
- **`fuzz/`** — libFuzzer harnesses with a committed corpus; `fuzz/AGENTS.md`.
