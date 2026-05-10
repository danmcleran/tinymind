# Embedded smoke test matrix.
#
# Exercises the (FLOAT, STD, QUANT) gate matrix to catch accidental
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

.PHONY: all embedded_smoke_test freestanding no_stdlib no_fpu hosted quant_freestanding run clean

# Default target keeps the historical name pointing at the matrix build, so
# `make` from the top-level driver still does the right thing.
all: freestanding no_stdlib no_fpu hosted quant_freestanding

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

run:
	$(OUT_FREESTANDING)
	$(OUT_NO_STDLIB)
	$(OUT_NO_FPU)
	$(OUT_HOSTED)
	$(OUT_QUANT_FREESTANDING)

clean:
	rm -f ./output/*
