# Fuzzing harnesses for TinyMind int8 compute kernels (libFuzzer + ASan/UBSan).
#
#   make fuzz        build + run ONE target interactively (TARGET=fuzz_qsoftmax)
#   make replay      build + replay the committed corpus only (-runs=0); fast,
#                    deterministic. This is the blocking PR gate -- a regression
#                    in any kernel re-triggers its archived crash input.
#   make fuzz-ci     build + run each target for MAX_TIME seconds (exploratory);
#                    nonzero exit on any crash / sanitizer error. Nightly cron.
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

TARGETS  = fuzz_qlayernorm fuzz_qsoftmax fuzz_qfft1d fuzz_requantize \
           fuzz_qattention_softmax fuzz_qmha fuzz_qlstm fuzz_qgru fuzz_qcfc \
           fuzz_simd_avx2_diff
MAX_TIME ?= 60
TARGET   ?= fuzz_qlayernorm

# Differential harness: AVX2 backend vs scalar reference in one binary.
# x86-only (every GitHub-hosted x64 runner has AVX2); on a non-x86 dev box
# drop it with `make TARGETS=fuzz_qlayernorm ...` or build targets singly.
fuzz_simd_avx2_diff : DEFINES += -DTINYMIND_ENABLE_SIMD_AVX2=1
fuzz_simd_avx2_diff : FUZZ_FLAGS += -mavx2

all : $(TARGETS)

# One pattern rule builds every harness; each is a single self-contained .cpp.
# Header-only library: every harness depends on the kernel headers, otherwise
# a header fix replays against a stale binary and "re-triggers" its own crash.
HEADERS = $(wildcard ../cpp/*.hpp) $(wildcard ../cpp/include/*.hpp)

fuzz_% : fuzz_%.cpp $(HEADERS)
	@command -v $(FUZZ_CXX) >/dev/null 2>&1 || { echo "ERROR: $(FUZZ_CXX) not found. Install Clang (sudo apt install clang) or pass FUZZ_CXX=..."; exit 1; }
	$(FUZZ_CXX) $(FUZZ_FLAGS) $(WARN) $(DEFINES) $(INCLUDES) -o $@ $<

# Interactive exploration of a single target until a crash or Ctrl-C.
fuzz : $(TARGET)
	@mkdir -p corpus/$(TARGET:fuzz_%=%)
	./$(TARGET) corpus/$(TARGET:fuzz_%=%)

# Deterministic regression gate: replay ONLY the committed corpus (-runs=0
# does no new mutation), so the result is reproducible and fast. Any archived
# crash that re-triggers means a kernel regressed. Empty corpus dirs are a
# no-op, so this stays green until the first crash is archived.
replay : $(TARGETS)
	@for t in $(TARGETS); do \
		d=corpus/$${t#fuzz_}; \
		mkdir -p $$d; \
		echo "=== replay: $$t ($$(ls -1 $$d | wc -l) inputs) ==="; \
		./$$t -runs=0 $$d || exit 1; \
	done
	@echo "replay: no regressions"

# Exploratory run: time-boxed, deterministic exit. Any crash/UB leaves a
# crash-* file and returns nonzero. Nightly cron, not the PR gate.
fuzz-ci : $(TARGETS)
	@for t in $(TARGETS); do \
		echo "=== fuzz-ci: $$t ($(MAX_TIME)s) ==="; \
		mkdir -p corpus/$${t#fuzz_}; \
		./$$t -max_total_time=$(MAX_TIME) -print_final_stats=1 corpus/$${t#fuzz_} || exit 1; \
	done
	@echo "fuzz-ci: no crashes"

clean :
	rm -f $(TARGETS) crash-* leak-* oom-* timeout-*

.PHONY : all fuzz replay fuzz-ci clean
