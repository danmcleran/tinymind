# unit_test

One Boost.Test suite per directory; each builds with `make clean && make && make run`
and needs `BOOST_HOME`. Repo-wide rules are in the root `AGENTS.md`; long-form
descriptions of each suite are in `docs/ARCHITECTURE.md`.

`nn/` (NN correctness), `qformat/` (QValue runtime + `static_assert`), `qformat_oracle/` (differential oracle vs an exact 128-bit reference; known-defect ratchet; operation chains; nightly seeded soak via `make oracle-soak`), `exhaustive/` (every input of small domains: LUT activations per format, requantizer fast paths, int8 dot per SIMD lane, softmax rows, layernorm sqrt; built `-O2`; `make full` = every int32, nightly), `qlearn/`, `quantization/` (int8 path), `embedded/` (cross-corner regression matrix + `simd_prereq_regressions`), `integration/` (Phase 16 golden-byte exemplar suite; int8 exemplars plus the Q16.16 ANFIS one), `pinn/`, `ltc/`, `cfc/`.

Suites not named above: `kan/` (KAN layers, including the fixed-point step-rounding case), `dual/` (forward-mode automatic differentiation: `Dual`, `MultiDual`, the dual activations), `lookuptable/` (the LUT interpolation helpers).

Rules that bite here:

- `nn/sgd_step_rounding_tests.hpp` and `nn/momentum_rounding_tests.hpp` pin the rounding of every SGD update site. A new trainable layer adds a case to both; planting a floored or toward-zero gradient step must fail a test.
- Under C++14 a test that uses a `static constexpr` member by reference (`BOOST_TEST(T::N == 3u)` binds by reference) must copy it to a local first.
- `embedded/` and `formal/` have their own `AGENTS.md`.
