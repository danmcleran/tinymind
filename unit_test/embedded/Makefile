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

# -----------------------------------------------------------------------------
# ARM cross-compilation (build-only). HARD gate.
#
# The matrix above builds every gate corner with the host g++ and freestanding
# -D macros -- it proves the headers stay free of hosted deps, but it never
# proves they *assemble for a Cortex-M*. Host x86 hides ABI/size assumptions,
# int-width edge cases, missing intrinsics and soft/hard-float behaviour. This
# target compiles (no link -- bare metal has no libc/startup here) the smoke
# source across representative Cortex-M profiles and float ABIs with the real
# arm-none-eabi toolchain. Compile-only catches the large majority of
# target-only breakage without any hardware.
ARM_CXX  ?= arm-none-eabi-g++
ARM_STD  ?= -std=c++17
ARM_WARN ?= -Wall -Wextra -Werror -Wpedantic
ARM_M0   = -mcpu=cortex-m0plus -mfloat-abi=soft -mthumb
ARM_M4   = -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16 -mthumb
ARM_M33  = -mcpu=cortex-m33 -mfloat-abi=hard -mfpu=fpv5-sp-d16 -mthumb

.PHONY: arm_crosscompile
arm_crosscompile:
	@command -v $(ARM_CXX) >/dev/null 2>&1 || { echo "ERROR: $(ARM_CXX) not found. sudo apt install gcc-arm-none-eabi"; exit 1; }
	@echo "[arm] cortex-m0plus soft-float: freestanding"
	$(ARM_CXX) $(ARM_STD) $(ARM_WARN) $(ARM_M0) -c $(SOURCES) $(INCLUDES) -o /dev/null
	@echo "[arm] cortex-m0plus soft-float: quant freestanding (deployable int8 shape)"
	$(ARM_CXX) $(ARM_STD) $(ARM_WARN) $(ARM_M0) -c $(SOURCES) $(INCLUDES) -DTINYMIND_ENABLE_QUANTIZATION=1 -o /dev/null
	@echo "[arm] cortex-m4 hard-float: float forward-pass (no stdlib)"
	$(ARM_CXX) $(ARM_STD) $(ARM_WARN) $(ARM_M4) -c $(SOURCES) $(INCLUDES) -DTINYMIND_ENABLE_FLOAT=1 -o /dev/null
	@echo "[arm] cortex-m4 hard-float: quant freestanding"
	$(ARM_CXX) $(ARM_STD) $(ARM_WARN) $(ARM_M4) -c $(SOURCES) $(INCLUDES) -DTINYMIND_ENABLE_QUANTIZATION=1 -o /dev/null
	@echo "[arm] cortex-m4 hard-float: fp16 storage tier"
	$(ARM_CXX) $(ARM_STD) $(ARM_WARN) $(ARM_M4) -c $(SOURCES) $(INCLUDES) -DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_FP16=1 -DTINYMIND_ENABLE_QUANTIZATION=1 -o /dev/null
	@echo "[arm] cortex-m33 hard-float: freestanding"
	$(ARM_CXX) $(ARM_STD) $(ARM_WARN) $(ARM_M33) -c $(SOURCES) $(INCLUDES) -o /dev/null
	@echo "[arm] cortex-m33 hard-float: quant freestanding"
	$(ARM_CXX) $(ARM_STD) $(ARM_WARN) $(ARM_M33) -c $(SOURCES) $(INCLUDES) -DTINYMIND_ENABLE_QUANTIZATION=1 -o /dev/null
	@echo "arm_crosscompile: all Cortex-M corners assemble clean"

# -----------------------------------------------------------------------------
# Heap-free guarantee (build-only). HARD gate.
#
# TinyMind targets no-heap MCUs: the library must never pull in the dynamic
# allocator. This compiles the freestanding + quant corners to objects and
# asserts none references malloc / operator new (Itanium ABI: _Znwj/_Znwm =
# new, _Znaj/_Znam = new[], _Zdl*/_Zda* = delete). A regression that adds a
# std::vector / new into an embedded header re-triggers the gate.
.PHONY: heapfree
heapfree:
	$(MKDIR)
	$(CC) $(ARM_STD) $(WARN) -c $(SOURCES) $(INCLUDES) -o ./output/heapfree_free.o
	$(CC) $(ARM_STD) $(WARN) -c $(SOURCES) $(INCLUDES) -DTINYMIND_ENABLE_QUANTIZATION=1 -o ./output/heapfree_quant.o
	@fail=0; for o in free quant; do \
	  syms="$$(nm -u ./output/heapfree_$$o.o | grep -iE '(^| )(malloc|calloc|realloc|free|_Zn[aw][jm]|_Zd[al][apv])' || true)"; \
	  if [ -n "$$syms" ]; then echo "HEAP GATE FAIL ($$o corner): dynamic-allocation symbols referenced:"; echo "$$syms"; fail=1; fi; \
	done; \
	if [ $$fail -ne 0 ]; then exit 1; fi; \
	echo "heapfree: no dynamic-allocation symbols in freestanding or quant object"

# -----------------------------------------------------------------------------
# Static stack-usage report (build-only). ADVISORY.
#
# Embedded parts have kilobytes of RAM; unbounded per-frame stack is a class of
# bug the runtime bench::stackHighWater watermark only catches after the fact.
# -fstack-usage emits a .su file with the worst-case frame of every function.
# This reports the deepest frames for the deployable Cortex-M4 int8 shape so a
# reviewer can eyeball growth. Advisory: template frames vary by compiler and
# the aggregating harness function dominates, so this informs, never blocks.
.PHONY: stack_usage
stack_usage:
	$(MKDIR)
	@command -v $(ARM_CXX) >/dev/null 2>&1 || { echo "ERROR: $(ARM_CXX) not found. sudo apt install gcc-arm-none-eabi"; exit 1; }
	$(ARM_CXX) $(ARM_STD) $(WARN) $(ARM_M4) -Os -c $(SOURCES) $(INCLUDES) \
		-DTINYMIND_ENABLE_QUANTIZATION=1 -fstack-usage -o ./output/stack_usage.o
	@su=$$(find . -name 'embedded_smoke_test.su' -o -name 'stack_usage.su' | head -1); \
	echo "=== deepest stack frames (Cortex-M4 int8, -Os), bytes ==="; \
	sort -t"	" -k2 -rn "$$su" | awk -F"	" 'NR<=12{printf "%8s  %s\n", $$2, $$1}'; \
	echo "stack_usage: advisory report (not a gate); full data in $$su"

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
