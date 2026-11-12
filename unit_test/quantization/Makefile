# Makefile for the quantization unit tests (Phase 1: qaffine.hpp).
#
# Quantization gate is on; FLOAT + STD enabled so quantizeMultiplier and
# its round-trip tests compile in.
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
SOURCES=quantization_unit_test.cpp
# TINYMIND_SPLIT_REQUANTIZER_FORMS defaults on only for Thumb-1 targets; on
# here so the host runs the dense kernels' split loops (see qaffine.hpp).
DEFINES=-DTINYMIND_ENABLE_QUANTIZATION=1 -DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_FP16=1 \
        -DTINYMIND_SPLIT_REQUANTIZER_FORMS=1
INCLUDES=-I../../include -I../../cpp -I../../cpp/include -I${BOOST_HOME}
OUT=./output/quantization_unit_test

quantization_unit_test:
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(DEFINES) $(INCLUDES)

run:
	cd ./output && ./quantization_unit_test && cd ../

coverage:
	$(MKDIR)
	$(CC) --coverage -O0 -g $(WARN) -o $(OUT) $(SOURCES) $(DEFINES) $(INCLUDES)

clean:
	rm -f ./output/*
