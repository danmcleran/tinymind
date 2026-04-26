# Simple Makefile for the lookup-table unit tests.
# Compiles cpp/lookupTables.cpp with TINYMIND_USE_* macros enabling every
# (function, Q-format) pair the test exercises.
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
SOURCES=lookuptable_unit_test.cpp ../../cpp/lookupTables.cpp
DEFINES=-DTINYMIND_ENABLE_OSTREAMS=1 \
        -DTINYMIND_USE_SIGMOID_8_8=1 -DTINYMIND_USE_SIGMOID_16_16=1 \
        -DTINYMIND_USE_TANH_8_8=1 -DTINYMIND_USE_TANH_16_16=1 \
        -DTINYMIND_USE_EXP_8_8=1 -DTINYMIND_USE_EXP_16_16=1 \
        -DTINYMIND_USE_LOG_8_8=1 -DTINYMIND_USE_LOG_16_16=1 \
        -DTINYMIND_USE_SIN_8_8=1 -DTINYMIND_USE_SIN_16_16=1 \
        -DTINYMIND_USE_COS_8_8=1 -DTINYMIND_USE_COS_16_16=1
INCLUDES=-I../../include -I../../cpp -I${BOOST_HOME}
OUT=./output/lookuptable_unit_test

lookuptable_unit_test:
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(DEFINES) $(INCLUDES)

run:
	cd ./output && ./lookuptable_unit_test && cd ../

clean:
	rm -f ./output/*
