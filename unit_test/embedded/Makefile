# Embedded smoke test.
#
# Compiles a QValue-only pipeline of every standalone composable layer
# WITHOUT defining TINYMIND_ENABLE_FLOAT, TINYMIND_ENABLE_HOSTED_IO, or
# TINYMIND_ENABLE_OSTREAMS. Catches accidental regressions where new
# code reintroduces an FPU- or stdlib-dependent path into headers used
# by embedded targets.
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
SOURCES=embedded_smoke_test.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include
OUT=./output/embedded_smoke_test

embedded_smoke_test:
	$(MKDIR)
	$(CC) -ggdb $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES)

run:
	./output/embedded_smoke_test

clean:
	rm -f ./output/*
