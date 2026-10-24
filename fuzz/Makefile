# Fuzzing harnesses for tinymind int8 compute kernels (libFuzzer + ASan/UBSan).
#
#   make fuzz        build + run ONE target interactively (TARGET=fuzz_qsoftmax)
#   make replay      build + replay the committed corpus only (-runs=0); fast,
#                    deterministic. This is the blocking PR gate -- a regression
#                    in any kernel re-triggers its archived crash input.
#   make fuzz-ci     build + run each target for MAX_TIME seconds (exploratory);
#                    nonzero exit on any crash / sanitizer error. Nightly cron.
#   make agree       replay the committed corpus through plain (unsanitized)
#                    builds from g++ and clang++ at -O0 and -O3 and require
#                    identical output digests from every build. PR gate.
#   make msan-replay replay the committed corpus under MemorySanitizer, which
#                    libFuzzer itself cannot host. PR gate.
#   make check-coverage
#                    require every TINYMIND_FUZZ_CHECK in every harness to
#                    execute at least once on the committed corpus. PR gate.
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
# crash that re-triggers means a kernel regressed, and every property check
# in the harnesses runs on every committed input. -timeout turns a hang into
# a failure; libFuzzer's default allows 20 minutes per input.
REPLAY_TIMEOUT ?= 10

replay : $(TARGETS)
	@for t in $(TARGETS); do \
		d=corpus/$${t#fuzz_}; \
		mkdir -p $$d; \
		echo "=== replay: $$t ($$(ls -1 $$d | wc -l) inputs) ==="; \
		./$$t -runs=0 -timeout=$(REPLAY_TIMEOUT) $$d || exit 1; \
	done
	@echo "replay: no regressions"

# Cross-build agreement. Each harness is linked with replay_main.cpp instead
# of libFuzzer, built without sanitizers by each compiler at each level, and
# run over its committed corpus; replay_main prints a digest of the outputs
# per input. Every build must print the same digests as the first one.
#
# A sanitizer build runs one compiler at one optimization level, so it cannot
# see undefined behaviour that is only exploited at -O3, or arithmetic whose
# result differs between compilers. This gate compares the outputs directly.
# The harnesses' property checks stay live in every build.
AGREE_GXX    ?= g++
AGREE_BUILDS ?= gcc-O0 gcc-O3 clang-O0 clang-O3
AGREE_FLAGS   = -std=c++17 $(WARN) -Werror $(DEFINES) $(INCLUDES)
AGREE_DIR     = agree

agree :
	@command -v $(AGREE_GXX) >/dev/null 2>&1 || { echo "ERROR: $(AGREE_GXX) not found."; exit 1; }
	@command -v $(FUZZ_CXX) >/dev/null 2>&1 || { echo "ERROR: $(FUZZ_CXX) not found. Install Clang or pass FUZZ_CXX=..."; exit 1; }
	@set -e; for b in $(AGREE_BUILDS); do \
		mkdir -p $(AGREE_DIR)/$$b; \
		case $$b in gcc-*) cxx="$(AGREE_GXX)" ;; *) cxx="$(FUZZ_CXX)" ;; esac; \
		opt="-$${b#*-}"; \
		for t in $(TARGETS); do \
			extra=""; [ $$t = fuzz_simd_avx2_diff ] && extra="-DTINYMIND_ENABLE_SIMD_AVX2=1 -mavx2"; \
			$$cxx $(AGREE_FLAGS) $$opt $$extra -o $(AGREE_DIR)/$$b/$$t $$t.cpp replay_main.cpp; \
			./$(AGREE_DIR)/$$b/$$t corpus/$${t#fuzz_} > $(AGREE_DIR)/$$b/$$t.txt; \
		done; \
		echo "=== agree: $$b built and replayed ==="; \
	done
	@set -e; base=$(firstword $(AGREE_BUILDS)); for t in $(TARGETS); do \
		n=$$(wc -l < $(AGREE_DIR)/$$base/$$t.txt); \
		for b in $(wordlist 2,99,$(AGREE_BUILDS)); do \
			diff -u $(AGREE_DIR)/$$base/$$t.txt $(AGREE_DIR)/$$b/$$t.txt > /dev/null || \
			{ echo "AGREE FAIL: $$t output differs between $$base and $$b:"; \
			  diff -u $(AGREE_DIR)/$$base/$$t.txt $(AGREE_DIR)/$$b/$$t.txt | head -20; exit 1; }; \
		done; \
		echo "agree: $$t ($$n inputs) identical across $(AGREE_BUILDS)"; \
	done

# MemorySanitizer over the committed corpus, through replay_main.cpp (the
# libFuzzer runtime is not MSan-instrumented on a stock toolchain, so the
# libFuzzer binaries cannot carry MSan). -O0 for the reason the top-level
# check-msan gives: at -O1 and above the optimizer folds an uninitialized
# read away before the instrumentation sees it.
MSAN_FLAGS = -fsanitize=memory -fsanitize-memory-track-origins=2 -fno-omit-frame-pointer \
             -O0 -g -std=c++17 $(WARN) $(DEFINES) $(INCLUDES)
MSAN_DIR   = msan

msan-replay :
	@command -v $(FUZZ_CXX) >/dev/null 2>&1 || { echo "ERROR: $(FUZZ_CXX) not found. Install Clang or pass FUZZ_CXX=..."; exit 1; }
	@mkdir -p $(MSAN_DIR)
	@set -e; for t in $(TARGETS); do \
		extra=""; [ $$t = fuzz_simd_avx2_diff ] && extra="-DTINYMIND_ENABLE_SIMD_AVX2=1 -mavx2"; \
		$(FUZZ_CXX) $(MSAN_FLAGS) $$extra -o $(MSAN_DIR)/$$t $$t.cpp replay_main.cpp; \
		./$(MSAN_DIR)/$$t corpus/$${t#fuzz_} > /dev/null || { echo "MSAN-REPLAY FAIL: $$t"; exit 1; }; \
		echo "msan-replay: $$t ($$(ls -1 corpus/$${t#fuzz_} | wc -l) inputs) clean"; \
	done

# Exploratory run: time-boxed, deterministic exit. Any crash/UB leaves a
# crash-* file and returns nonzero. Nightly cron, not the PR gate.
fuzz-ci : $(TARGETS)
	@for t in $(TARGETS); do \
		echo "=== fuzz-ci: $$t ($(MAX_TIME)s) ==="; \
		mkdir -p corpus/$${t#fuzz_}; \
		./$$t -max_total_time=$(MAX_TIME) -print_final_stats=1 corpus/$${t#fuzz_} || exit 1; \
	done
	@echo "fuzz-ci: no crashes"

# Property-check reachability. A check the committed corpus never executes
# guards nothing in the replay gate -- e.g. an accuracy check behind a
# "no saturation possible" guard that no committed input satisfies. Each
# harness is built with gcov instrumentation, replayed over its corpus, and
# every line holding a TINYMIND_FUZZ_CHECK must show a nonzero count. Checks
# sit on their own lines so a line count is the check's count.
COV_DIR = coverage

check-coverage :
	@command -v $(AGREE_GXX) >/dev/null 2>&1 || { echo "ERROR: $(AGREE_GXX) not found."; exit 1; }
	@mkdir -p $(COV_DIR)
	@set -e; $(AGREE_GXX) -std=c++17 -O0 $(WARN) -Werror $(INCLUDES) -c replay_main.cpp -o $(COV_DIR)/replay_main.o; \
	for t in $(TARGETS); do \
		extra=""; [ $$t = fuzz_simd_avx2_diff ] && extra="-DTINYMIND_ENABLE_SIMD_AVX2=1 -mavx2"; \
		rm -f $(COV_DIR)/$$t.gcda; \
		$(AGREE_GXX) --coverage -std=c++17 -O0 $(WARN) -Werror $(DEFINES) $(INCLUDES) $$extra -c $$t.cpp -o $(COV_DIR)/$$t.o; \
		$(AGREE_GXX) --coverage $(COV_DIR)/$$t.o $(COV_DIR)/replay_main.o -o $(COV_DIR)/$$t; \
		./$(COV_DIR)/$$t corpus/$${t#fuzz_} > /dev/null; \
		gcov -o $(COV_DIR) $$t.cpp > /dev/null; mv -f *.gcov $(COV_DIR)/; \
		total=$$(grep -c 'TINYMIND_FUZZ_CHECK(' $$t.cpp); \
		[ $$total -gt 0 ] || { echo "CHECK-COVERAGE FAIL: $$t has no property checks"; exit 1; }; \
		dead=$$(grep -E '^ *(#####|=====): *[0-9]+:.*TINYMIND_FUZZ_CHECK\(' $(COV_DIR)/$$t.cpp.gcov || true); \
		if [ -n "$$dead" ]; then \
			echo "CHECK-COVERAGE FAIL: $$t has property checks the committed corpus never executes:"; \
			echo "$$dead"; exit 1; \
		fi; \
		echo "check-coverage: $$t ($$total checks) all executed by the committed corpus"; \
	done

clean :
	rm -f $(TARGETS) crash-* leak-* oom-* timeout-*
	rm -rf $(AGREE_DIR) $(MSAN_DIR) $(COV_DIR)

.PHONY : all fuzz replay fuzz-ci agree msan-replay check-coverage clean
