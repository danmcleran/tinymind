# Phase 16 integration tests. Boost.Test fixture per exemplar that
# shells out to the example binary with --golden and verifies the
# emitted int8 byte stream matches the baked-in expected string
# byte-for-byte. Locks regressions across SIMD gate combos.
#
# Requires the four exemplar binaries to be built first. Root Makefile
# orders this target after the example targets.

MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
SOURCES=integration_unit_test.cpp
DEFINES=
INCLUDES=-I../../include -I../../cpp -I../../cpp/include -I${BOOST_HOME}
OUT=./output/integration_unit_test

integration_unit_test:
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(DEFINES) $(INCLUDES)

run:
	cd ./output && ./integration_unit_test && cd ../

clean:
	rm -f ./output/*
