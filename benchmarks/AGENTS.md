# benchmarks

Repo-wide rules are in the root `AGENTS.md`.

On-target comparisons against other frameworks, kept outside `examples/` because they target Arm only (the top-level build and `check-clang-examples` compile every `examples/` Makefile with the host compiler); not built in CI. `framework_comparison_f411/`: tinymind vs TFLM (CMSIS-NN/reference) vs emlearn on a NUCLEO-F411RE, int8 outputs bit-exact with TFLM; `make deps` fetches the venv + TFLM, `make LEVEL=O3 run` flashes and captures. `make host-bench` times the same models on the build host, scalar vs each x86 SIMD backend the CPU supports. `SUITE=mlperf` (F411 only) runs the MLPerf Tiny reference KWS and ResNet-8 int8 models against TFLM instead; `make mlperf-models` fetches them (Apache-2.0, never committed: `gen/mlperf/` is gitignored), and results must not be called MLPerf results (no runner, unverified). Write-up: `docs/benchmarks.md`.

## Benchmark harness (`cpp/include/bench/`)

- **`platform.hpp`** — `bench::readCycleCounter()` (DWT CYCCNT on Cortex-M, `std::chrono` host fallback) and `bench::paintStack` / `bench::stackHighWater` for MCU stack watermarking. Enable the MCU path with `-DTINYMIND_BENCH_CORTEX_M`.
- **`report.hpp`** — `bench::LayerStat` CSV row type, `bench::writeHeader` / `bench::writeRow` for any sink that supports `operator<<`, and a `bench::ScopedTimer` for per-layer measurements. See `examples/kws_cortex_m/` for an end-to-end use.

Per-layer numbers are only trustworthy when each layer is isolated with `noinline`; otherwise the compiler merges layers and the split is meaningless.
