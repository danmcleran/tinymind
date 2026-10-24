# Makefile for the exhaustive property suite (exhaustive_unit_test.cpp).
#
# Each case checks a property on every element of a stated domain, so the
# suite is built optimized: at -O0 the int8 sweeps alone take minutes. The
# sanitizer and second-toolchain targets override CC; -O2 here still applies.
#
#   make && make run   PR run (part of `make check`): the domains below plus,
#                      for the int32 accumulator sweeps, the range an int8
#                      layer reaches (|acc| <= 2^22).
#   make full          nightly: every int32 accumulator and every uint32
#                      square-root input, parallelized with OpenMP.
#
# The LUT case needs every activation table it sweeps compiled in.
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
SOURCES=exhaustive_unit_test.cpp ../../cpp/lookupTables.cpp
LUT_FORMATS=4_4 5_3 4_12 5_11 6_10 7_9 8_8 9_7 10_6 11_5 12_4 13_3 8_24 16_16 24_8
LUT_DEFINES=$(foreach f,$(LUT_FORMATS),-DTINYMIND_USE_SIGMOID_$(f)=1 -DTINYMIND_USE_TANH_$(f)=1)
DEFINES=-DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_QUANTIZATION=1 $(LUT_DEFINES)
INCLUDES=-I../../include -I../../cpp -I../../cpp/include -I${BOOST_HOME}
OUT=./output/exhaustive_unit_test

exhaustive_unit_test:
	$(MKDIR)
	$(CC) -g -O2 $(WARN) -o $(OUT) $(SOURCES) $(DEFINES) $(INCLUDES)

run:
	cd ./output && ./exhaustive_unit_test && cd ../

full:
	$(MKDIR)
	$(CC) -g -O2 -fopenmp $(WARN) -o $(OUT)_full $(SOURCES) $(DEFINES) $(INCLUDES)
	cd ./output && TINYMIND_EXHAUSTIVE_FULL=1 ./exhaustive_unit_test_full && cd ../

coverage:
	$(MKDIR)
	$(CC) --coverage -O0 -g $(WARN) -o $(OUT) $(SOURCES) $(DEFINES) $(INCLUDES)

clean:
	rm -f ./output/*
