# Phase 14 perf matrix example.
#
# Builds the same int8 block under several SIMD gate combinations on the
# build host (x86 here) and runs each binary to emit a CSV row. The
# checksum column doubles as a bit-exactness regression: a non-scalar
# row with a different checksum than the scalar row indicates a broken
# SIMD backend.
#
# Cross-compiled backends (NEON, NEON-DOTPROD, SVE, Helium MVE) are not
# built here by default — they need the matching toolchain. Add a
# target per cross-compiler as needed.

MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
SOURCES=perf_matrix.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_QUANTIZATION=1
OPT=-O3

OUT_SCALAR=./output/perf_matrix_scalar
OUT_AVX2=./output/perf_matrix_avx2
OUT_AVX512F=./output/perf_matrix_avx512f
OUT_AVX512_VNNI=./output/perf_matrix_avx512_vnni
OUT_REPORT=./output/perf_report.csv

.PHONY: default all scalar avx2 avx512f avx512_vnni report clean

default: scalar avx2 avx512f avx512_vnni
all:     default

scalar:
	$(MKDIR)
	$(CC) $(OPT) $(WARN) -o $(OUT_SCALAR) $(SOURCES) $(INCLUDES) $(DEFINES)

avx2:
	$(MKDIR)
	$(CC) $(OPT) $(WARN) -mavx2 -mssse3 -o $(OUT_AVX2) $(SOURCES) $(INCLUDES) $(DEFINES) -DTINYMIND_ENABLE_SIMD_AVX2=1

avx512f:
	$(MKDIR)
	$(CC) $(OPT) $(WARN) -mavx512f -mavx512bw -o $(OUT_AVX512F) $(SOURCES) $(INCLUDES) $(DEFINES) -DTINYMIND_ENABLE_SIMD_AVX512F=1

avx512_vnni:
	$(MKDIR)
	$(CC) $(OPT) $(WARN) -mavx512f -mavx512bw -mavx512vnni -o $(OUT_AVX512_VNNI) $(SOURCES) $(INCLUDES) $(DEFINES) -DTINYMIND_ENABLE_SIMD_AVX512F=1 -DTINYMIND_ENABLE_SIMD_AVX512_VNNI=1

# Run every binary that exists and concatenate their CSV rows. Header is
# emitted once.
report: default
	@$(MKDIR)
	@true > $(OUT_REPORT)
	@printed_header=0; \
	for b in $(OUT_SCALAR) $(OUT_AVX2) $(OUT_AVX512F) $(OUT_AVX512_VNNI); do \
	    if [ -x $$b ]; then \
	        out=$$($$b); \
	        if [ $$printed_header -eq 0 ]; then \
	            echo "$$out" >> $(OUT_REPORT); \
	            printed_header=1; \
	        else \
	            echo "$$out" | tail -n +2 >> $(OUT_REPORT); \
	        fi; \
	    fi; \
	done
	@cat $(OUT_REPORT)

run: report

clean:
	rm -f ./output/*
