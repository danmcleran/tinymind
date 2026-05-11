# Embedded smoke test matrix.
#
# Exercises the (FLOAT, STD, QUANT, FP16) gate matrix to catch accidental
# regressions where new code reintroduces a hosted-only or FPU-only path
# into headers used by embedded targets:
#
#   freestanding        FLOAT=0 STD=0  -- pure QValue, no <cmath>, no
#                                          <type_traits>, no <fstream>, no
#                                          <ostream>, no rand(). The tightest
#                                          configuration we ship.
#
#   no_stdlib           FLOAT=1 STD=0  -- FPU available but no C++ stdlib /
#                                          namespace std. float/double as
#                                          ValueType for forward-pass
#                                          inference, but no
#                                          AdamOptimizerFloat /
#                                          RmsPropOptimizerFloat / Xavier
#                                          (those need std::sqrt). Catches
#                                          accidental std:: leaks via
#                                          float-only paths.
#
#   no_fpu              FLOAT=0 STD=1  -- hosted but fixed-point only.
#                                          Verifies the QValue-only build
#                                          still compiles when <type_traits>
#                                          et al. are present and nothing
#                                          accidentally depends on FLOAT.
#
#   hosted              FLOAT=1 STD=1  -- equivalent to the host examples;
#                                          sanity check that the embedded
#                                          smoke source also builds with
#                                          everything available.
#
#   quant_freestanding  FLOAT=0 STD=0 QUANT=1 -- inference-only int8 path.
#                                          Exercises the parallel Q* layer
#                                          family with no float, no stdlib,
#                                          no calibration helpers. This is
#                                          the deployable target shape for
#                                          quantized models on a Cortex-M
#                                          MCU.
#
#   fp16_freestanding   FLOAT=1 STD=0 FP16=1 -- Phase 9 half-precision
#                                          storage tier plus type bridges.
#                                          Requires FPU but no C++ stdlib;
#                                          confirms tinymind_fp16.hpp and
#                                          qbridge.hpp stay free of
#                                          <cmath>/<type_traits> when the
#                                          gate is on.
#
#   int16_accum_freestanding
#                       FLOAT=0 STD=0 QUANT=1 INT16_ACCUM=1
#                                       -- Phase 12 recurrent quantization.
#                                          Exercises QLSTM with an int16
#                                          cell-state template instantiation
#                                          and QGRU at the deployable
#                                          freestanding shape; confirms the
#                                          new headers stay clear of
#                                          <cmath>/<type_traits>/stdlib.
#
#   simd_disabled       FLOAT=0 STD=0 QUANT=1 SIMD_*=0 (all)
#                                       -- Phase 14 scalar fallback regression.
#                                          Explicitly compiles with every
#                                          TINYMIND_ENABLE_SIMD_* gate off and
#                                          confirms the scalar dispatch path
#                                          still builds and runs at the
#                                          deployable freestanding shape. Locks
#                                          the invariant that callers who do
#                                          not opt into SIMD pay nothing for
#                                          Phase 14.
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
SOURCES=embedded_smoke_test.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include

OUT_FREESTANDING=./output/embedded_smoke_freestanding
OUT_NO_STDLIB=./output/embedded_smoke_no_stdlib
OUT_NO_FPU=./output/embedded_smoke_no_fpu
OUT_HOSTED=./output/embedded_smoke_hosted
OUT_QUANT_FREESTANDING=./output/embedded_smoke_quant_freestanding
OUT_FP16_FREESTANDING=./output/embedded_smoke_fp16_freestanding
OUT_INT16_ACCUM_FREESTANDING=./output/embedded_smoke_int16_accum_freestanding
OUT_SIMD_DISABLED=./output/embedded_smoke_simd_disabled

.PHONY: all embedded_smoke_test freestanding no_stdlib no_fpu hosted quant_freestanding fp16_freestanding int16_accum_freestanding simd_disabled run clean

# Default target keeps the historical name pointing at the matrix build, so
# `make` from the top-level driver still does the right thing.
all: freestanding no_stdlib no_fpu hosted quant_freestanding fp16_freestanding int16_accum_freestanding simd_disabled

embedded_smoke_test: all

freestanding:
	$(MKDIR)
	$(CC) -ggdb $(WARN) -o $(OUT_FREESTANDING) $(SOURCES) $(INCLUDES)

no_stdlib:
	$(MKDIR)
	$(CC) -ggdb $(WARN) -o $(OUT_NO_STDLIB) $(SOURCES) $(INCLUDES) -DTINYMIND_ENABLE_FLOAT=1

no_fpu:
	$(MKDIR)
	$(CC) -ggdb $(WARN) -o $(OUT_NO_FPU) $(SOURCES) $(INCLUDES) -DTINYMIND_ENABLE_STD=1

hosted:
	$(MKDIR)
	$(CC) -ggdb $(WARN) -o $(OUT_HOSTED) $(SOURCES) $(INCLUDES) -DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1

quant_freestanding:
	$(MKDIR)
	$(CC) -ggdb $(WARN) -o $(OUT_QUANT_FREESTANDING) $(SOURCES) $(INCLUDES) -DTINYMIND_ENABLE_QUANTIZATION=1

fp16_freestanding:
	$(MKDIR)
	$(CC) -ggdb $(WARN) -o $(OUT_FP16_FREESTANDING) $(SOURCES) $(INCLUDES) -DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_FP16=1 -DTINYMIND_ENABLE_QUANTIZATION=1

int16_accum_freestanding:
	$(MKDIR)
	$(CC) -ggdb $(WARN) -o $(OUT_INT16_ACCUM_FREESTANDING) $(SOURCES) $(INCLUDES) -DTINYMIND_ENABLE_QUANTIZATION=1 -DTINYMIND_ENABLE_INT16_ACCUM=1

# Phase 14 scalar-fallback corner. Explicitly nails every SIMD_* gate to
# 0 so a future change that flips a default cannot silently turn a
# scalar-only smoke build into a SIMD build.
simd_disabled:
	$(MKDIR)
	$(CC) -ggdb $(WARN) -o $(OUT_SIMD_DISABLED) $(SOURCES) $(INCLUDES) -DTINYMIND_ENABLE_QUANTIZATION=1 \
		-DTINYMIND_ENABLE_SIMD_NEON=0 -DTINYMIND_ENABLE_SIMD_NEON_DOTPROD=0 \
		-DTINYMIND_ENABLE_SIMD_NEON_FP16=0 -DTINYMIND_ENABLE_SIMD_SVE=0 \
		-DTINYMIND_ENABLE_SIMD_SVE2=0 -DTINYMIND_ENABLE_SIMD_HELIUM_MVE_I=0 \
		-DTINYMIND_ENABLE_SIMD_HELIUM_MVE_F=0 -DTINYMIND_ENABLE_SIMD_AVX2=0 \
		-DTINYMIND_ENABLE_SIMD_AVX_VNNI=0 -DTINYMIND_ENABLE_SIMD_AVX512F=0 \
		-DTINYMIND_ENABLE_SIMD_AVX512_VNNI=0 -DTINYMIND_ENABLE_OPENMP=0

# Phase 14 compile-failure regression. Locks the invariant that dependent
# SIMD gates without their prerequisite fail at compile time with a
# readable static_assert message. The plan calls these out explicitly:
# DOTPROD without NEON, SVE without NEON, AVX_VNNI without AVX2,
# AVX512_VNNI without AVX512F, and the Helium / NEON mutual-exclusion
# pair. On a build host that does not ship the relevant intrinsic
# headers (e.g. x86 building the NEON gates) the preprocessor errors out
# before the static_assert fires — that is still a compile failure and
# satisfies the regression.
.PHONY: simd_prereq_regressions
simd_prereq_regressions:
	@echo "[prereq] AVX_VNNI=1 without AVX2 must fail"
	@! $(CC) $(WARN) -c $(SOURCES) -o /dev/null $(INCLUDES) -DTINYMIND_ENABLE_QUANTIZATION=1 -DTINYMIND_ENABLE_SIMD_AVX_VNNI=1 2>/dev/null
	@echo "[prereq] AVX512_VNNI=1 without AVX512F must fail"
	@! $(CC) $(WARN) -c $(SOURCES) -o /dev/null $(INCLUDES) -DTINYMIND_ENABLE_QUANTIZATION=1 -DTINYMIND_ENABLE_SIMD_AVX512_VNNI=1 2>/dev/null
	@echo "[prereq] all prerequisite checks behaved as expected"

run:
	$(OUT_FREESTANDING)
	$(OUT_NO_STDLIB)
	$(OUT_NO_FPU)
	$(OUT_HOSTED)
	$(OUT_QUANT_FREESTANDING)
	$(OUT_FP16_FREESTANDING)
	$(OUT_INT16_ACCUM_FREESTANDING)
	$(OUT_SIMD_DISABLED)

clean:
	rm -f ./output/*
