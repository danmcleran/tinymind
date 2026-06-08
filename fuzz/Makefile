# Fuzzing harnesses for TinyMind int8 compute kernels (libFuzzer + ASan/UBSan).
#
#   make fuzz        build + run interactively (until a crash or Ctrl-C)
#   make fuzz-ci     build + run each target for MAX_TIME seconds; nonzero
#                    exit on any crash / sanitizer error (for CI)
#   make clean
#
# libFuzzer requires Clang. Use a dedicated variable (not CXX, which the
# environment / parent make often pins to g++, which has no -fsanitize=fuzzer).
# The box has no unversioned `clang++`; override with `make FUZZ_CXX=clang++`.
FUZZ_CXX ?= clang++-18

FUZZ_SANITIZE = -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all
FUZZ_FLAGS    = $(FUZZ_SANITIZE) -O1 -g -std=c++17
WARN          = -Wall -Wextra -Wpedantic
DEFINES       = -DTINYMIND_ENABLE_QUANTIZATION=1 -DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1
INCLUDES      = -I../cpp -I../cpp/include -I../include

TARGETS  = fuzz_qlayernorm
MAX_TIME ?= 60

all : $(TARGETS)

fuzz_qlayernorm : fuzz_qlayernorm.cpp
	@command -v $(FUZZ_CXX) >/dev/null 2>&1 || { echo "ERROR: $(FUZZ_CXX) not found. Install Clang (sudo apt install clang) or pass CXX=..."; exit 1; }
	$(FUZZ_CXX) $(FUZZ_FLAGS) $(WARN) $(DEFINES) $(INCLUDES) -o $@ $<

# Interactive run: explores until it finds a crash or you Ctrl-C.
fuzz : $(TARGETS)
	@mkdir -p corpus/qlayernorm
	./fuzz_qlayernorm corpus/qlayernorm

# CI run: time-boxed, deterministic exit. Any crash/UB leaves a crash-* file
# and returns nonzero.
fuzz-ci : $(TARGETS)
	@for t in $(TARGETS); do \
		echo "=== fuzz-ci: $$t ($(MAX_TIME)s) ==="; \
		mkdir -p corpus/$${t#fuzz_}; \
		./$$t -max_total_time=$(MAX_TIME) -print_final_stats=1 corpus/$${t#fuzz_} || exit 1; \
	done
	@echo "fuzz-ci: no crashes"

clean :
	rm -f $(TARGETS) crash-* leak-* oom-* timeout-*
	rm -rf corpus

.PHONY : all fuzz fuzz-ci clean
