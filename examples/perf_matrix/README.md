# perf_matrix

Phase 14 SIMD gate benchmark. Builds the same int8 MobileNetV2-shaped
block (`QConv2D` 3x3 + `QDense`) under several `TINYMIND_ENABLE_SIMD_*`
gate combinations and emits a single CSV per binary:

```
active_backend,conv_iters,conv_total_us,conv_us_per_call,
dense_iters,dense_total_us,dense_us_per_call,
conv_output_checksum,dense_output_checksum
```

The `*_output_checksum` columns are the same across backends when Phase
14's bit-exactness invariant holds — any disagreement is a backend bug.

## Usage

```sh
make            # builds scalar / avx2 / avx512f / avx512_vnni
make report     # runs every binary and prints the combined CSV
make clean
```

`make report` writes `output/perf_report.csv` with one row per backend.

## Cross-arch backends

The default Makefile only builds the x86 backends that the host
compiler can target. To bench an Arm gate, add a target with the
matching cross-compiler, e.g.:

```make
neon_dotprod:
	$(MKDIR)
	aarch64-linux-gnu-g++ $(OPT) $(WARN) -march=armv8.2-a+dotprod \
	  -o ./output/perf_matrix_neon_dotprod $(SOURCES) $(INCLUDES) \
	  $(DEFINES) -DTINYMIND_ENABLE_SIMD_NEON=1 \
	  -DTINYMIND_ENABLE_SIMD_NEON_DOTPROD=1
```

Run the resulting binary on the target hardware (or under qemu-aarch64
for correctness checks).

## Notes on the checksum

`output_checksum` mixes a running sum with an FNV-style XOR-and-shift
hash of every output byte. It is order-sensitive in the hash term and
value-sensitive in both terms, so a one-byte off-by-one in any backend
flips the value visibly.

The reference (`scalar` row) is the source of truth.
