# Makefile to build and run all unit tests, build examples, and build apps.
default :
	@echo "Use 'make check' to build and run all unit tests."
check :
	cd unit_test/nn && make clean && make && make run && cd -
	cd unit_test/qformat && make clean && make && make run && cd -
	cd unit_test/qformat_oracle && make clean && make && make run && cd -
	cd unit_test/qlearn && make clean && make && make run && cd -
	cd unit_test/lookuptable && make clean && make && make run && cd -
	cd unit_test/embedded && make clean && make && make run && make simd_prereq_regressions && make qtree_static_regressions && cd -
	cd unit_test/quantization && make clean && make && make run && cd -
	cd unit_test/dual && make clean && make && make run && cd -
	cd unit_test/pinn && make clean && make && make run && cd -
	cd unit_test/ltc && make clean && make && make run && cd -
	cd unit_test/cfc && make clean && make && make run && cd -
	cd examples/pinn_heat1d && make clean && make && make release && make run && make train && cd -
	cd examples/ltc_sequence && make clean && make && make release && make run && cd -
	cd examples/cfc_sequence && make clean && make && make release && make run && cd -
	cd examples/anfis_mackey_glass && make clean && make && make release && make run && make golden && cd -
	cd examples/anfis_mackey_glass_int8 && make clean && make && make release && make run && make golden && cd -
	cd examples/qcfc_liquid_int8 && make clean && make && make release && make run && make bench && make golden && cd -
	cd examples/xor && make clean && make && make release && cd -
	cd examples/maze && make clean && make && make release && cd -
	cd examples/dqn_maze && make clean && make && make release && cd -
	cd unit_test/kan && make clean && make && make run && cd -
	cd examples/kan_xor && make clean && make && make release && cd -
	cd examples/kws_cortex_m && make clean && make && make release && cd -
	cd examples/kws_cortex_m_int8 && make clean && make && make release && cd -
	cd examples/resnet_block_int8 && make clean && make && make release && make run && cd -
	cd examples/transformer_encoder_int8 && make clean && make && make release && make run && cd -
	cd examples/transformer_encoder_stack_int8 && make clean && make && make release && make run && make golden && cd -
	cd examples/transformer_encoder_stack_softmax_int8 && make clean && make && make release && make run && make golden && cd -
	cd examples/seq2seq_int8 && make clean && make && make release && make run && make golden && cd -
	cd examples/seq2seq_softmax_int8 && make clean && make && make release && make run && make golden && cd -
	cd examples/tiny_generate_int8 && make clean && make && make release && make run && make golden && cd -
	cd examples/state_space_int8 && make clean && make && make release && make run && make golden && cd -
	cd examples/gbdt_tabular_int8 && make clean && make && make release && make run && make golden && cd -
	cd examples/moe_regimes_int8 && make clean && make && make release && make run && make bench && make golden && cd -
	cd examples/resnet18_block_int8 && make clean && make && make release && make run && cd -
	cd examples/mobilenetv2_int8 && make clean && make && make release && make run && cd -
	cd examples/mixed_precision_kws && make clean && make && make release && make run && cd -
	cd examples/mixed_precision_mlp_int8_qformat && make clean && make && make release && make run && cd -
	cd unit_test/integration && make clean && make && make run && cd -
	cd examples/pytorch_quant/xor && make clean && make && make release && make run && cd -
	cd examples/import_demo && make clean && make && make release && make run && cd -
	cd examples/import_moe_demo && make clean && make && make release && make run && cd -
	cd examples/iris && make clean && make release && make run && cd -
	cd examples/energy_efficiency && make clean && make release && make run && cd -
	cd examples/optical_digits && make clean && make release && make run && cd -
	cd examples/har_activity && make clean && make release && make run && cd -
	cd examples/gas_sensor_drift && make clean && make release && make run && cd -
	cd examples/air_quality && make clean && make release && make run && cd -
	cd examples/lstm_sinusoid_float && make clean && make release && make run && cd -
	cd examples/elman_temporal_xor && make clean && make release && make run && cd -
	cd examples/elman_vowels && make clean && make release && make run && cd -
	cd examples/perf_matrix && make clean && make && make report && cd -
	cd apps/activation && make clean && make && make release && cd -

# Code coverage (gcov + lcov). Requires lcov: sudo apt install lcov
# Instruments the runtime Boost suites plus the int8 example inference path,
# runs them, then aggregates line coverage of the cpp/ library into HTML.
# Compile-time-only suites (qformat, embedded) are excluded on purpose:
# gcov records executed arcs, and those suites verify via static_assert /
# compile-success, so they contribute no runtime coverage signal.
COV_SUITES = unit_test/nn unit_test/qlearn unit_test/quantization \
             unit_test/kan unit_test/lookuptable unit_test/dual unit_test/ltc \
             unit_test/cfc unit_test/pinn
COV_EXAMPLES = examples/resnet_block_int8 examples/resnet18_block_int8 \
               examples/mobilenetv2_int8 examples/transformer_encoder_int8 \
               examples/kws_cortex_m_int8 examples/mixed_precision_mlp_int8_qformat

LCOV_IGNORE = --ignore-errors mismatch,negative,source,unused,empty,gcov,inconsistent
# genhtml rejects the lcov-only 'gcov' error class; keep its own list.
GENHTML_IGNORE = --ignore-errors mismatch,negative,source,unused,empty,inconsistent

coverage : coverage-clean
	@command -v lcov >/dev/null 2>&1 || { echo "ERROR: lcov not found. Install with: sudo apt install lcov"; exit 1; }
	mkdir -p coverage
	for d in $(COV_SUITES) $(COV_EXAMPLES); do \
		echo "=== coverage: $$d ==="; \
		( cd $$d && make clean && make coverage && make run ) || exit 1; \
	done
	# Capture per suite: source paths are recorded relative to each suite's
	# compile dir, so --base-directory must point at that suite (gcov data
	# lives one level deeper in output/). Merge the per-suite .info files.
	# Re-assert coverage/ here -- it must survive the build loop, but a
	# concurrent `make clean`/`coverage-clean` (rm -rf coverage) or an
	# interrupted prior run can remove it, which would otherwise fail the
	# redirect below with "Directory nonexistent".
	mkdir -p coverage
	: > coverage/merge_args
	for d in $(COV_SUITES) $(COV_EXAMPLES); do \
		out=coverage/$$(echo $$d | tr / _).info; \
		lcov --capture --directory $$d --base-directory $$d \
		     --output-file $$out --rc geninfo_unexecuted_blocks=1 \
		     --rc branch_coverage=1 $(LCOV_IGNORE) || true; \
		[ -s $$out ] && echo "-a $$out" >> coverage/merge_args; \
	done
	lcov $$(cat coverage/merge_args) --output-file coverage/all.info --rc branch_coverage=1 $(LCOV_IGNORE)
	lcov --extract coverage/all.info '*/cpp/*' --rc branch_coverage=1 \
	     --output-file coverage/tinymind.info $(LCOV_IGNORE)
	genhtml coverage/tinymind.info --output-directory coverage/html $(GENHTML_IGNORE)
	python3 tools/coverage_dashboard.py coverage/tinymind.info coverage/dashboard.html
	@echo "Dashboard:   coverage/dashboard.html"
	@echo "HTML report: coverage/html/index.html"

# Coverage regression gate. Fails if cpp/ line coverage drops below
# COVERAGE_MIN_LINES or function coverage below COVERAGE_MIN_FUNCS. Functions
# sit at 100% (every template entry point is instantiated and run); lines sit
# at ~99.9% -- a handful of defensive edge guards (e.g. sqrt-of-non-positive,
# activation-table saturation) are not hit by the current fixed-input tests.
# Branch coverage is captured and reported but not gated -- gcov branch counts
# are noisy across template instantiations and compiler versions.
# Override on the command line to tighten, e.g. `make coverage-check COVERAGE_MIN_LINES=100`.
COVERAGE_MIN_LINES ?= 99.0
COVERAGE_MIN_FUNCS ?= 100.0
coverage-check :
	@[ -f coverage/tinymind.info ] || { echo "ERROR: no capture. Run 'make coverage' first."; exit 1; }
	@lcov --summary coverage/tinymind.info --rc branch_coverage=1 $(LCOV_IGNORE) 2>/dev/null \
	     | tee coverage/summary.txt
	@awk -v lmin=$(COVERAGE_MIN_LINES) -v fmin=$(COVERAGE_MIN_FUNCS) ' \
	    /lines[.]+:/     { l=$$2+0 } \
	    /functions[.]+:/ { f=$$2+0 } \
	    END { \
	      printf "coverage-check: lines=%.1f%% (floor %.1f%%)  functions=%.1f%% (floor %.1f%%)\n", l, lmin, f, fmin; \
	      if (l < lmin || f < fmin) { print "COVERAGE GATE: FAIL"; exit 1 } \
	      print "COVERAGE GATE: PASS" \
	    }' coverage/summary.txt

# Regenerate just the dashboard from an existing capture (no rebuild/re-run).
coverage-dashboard :
	python3 tools/coverage_dashboard.py coverage/tinymind.info coverage/dashboard.html
	@echo "Dashboard:   coverage/dashboard.html"

# Open the coverage dashboard in the user's default browser. Generate it first
# with `make coverage`. Honors $BROWSER, then the platform opener (xdg-open on
# Linux, open on macOS, start on Windows); no specific browser is assumed.
coverage-open :
	@[ -f coverage/dashboard.html ] || { echo "No dashboard. Run 'make coverage' first."; exit 1; }
	@f="$(CURDIR)/coverage/dashboard.html"; \
	if [ -n "$$BROWSER" ]; then "$$BROWSER" "$$f"; \
	elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$$f"; \
	elif command -v open >/dev/null 2>&1; then open "$$f"; \
	elif command -v start >/dev/null 2>&1; then start "" "$$f"; \
	else echo "Could not detect a browser opener. Open it manually: $$f"; fi

coverage-clean :
	find unit_test examples cpp -name '*.gcno' -delete 2>/dev/null || true
	find unit_test examples cpp -name '*.gcda' -delete 2>/dev/null || true
	rm -rf coverage

# =============================================================================
# Static & dynamic analysis
#
#   make sanitize   HARD GATE  -- rebuild every runtime suite + int8 example
#                               with ASan+UBSan and run; any UB aborts nonzero.
#   make cppcheck   HARD GATE  -- cppcheck over cpp/ (warning + portability).
#   make misra      ADVISORY   -- cppcheck MISRA C:2012 addon over cpp/. This is
#                               the MISRA *C* ruleset run against C++17 template
#                               code, so most findings are expected/informational
#                               (MISRA C++ needs a commercial qualified tool).
#                               Reports to misra-report.txt; pass
#                               MISRA_RULE_TEXTS=<file> for full rule descriptions
#                               instead of rule IDs.
#   make tidy       ADVISORY   -- clang-tidy via a bear-captured compile DB;
#                               reports to tidy-report.txt, never fails. Its
#                               clang-analyzer-* checks cover the clang static
#                               analyzer; CodeQL runs the same engine in CI.
#   make analyze               -- cppcheck (gate) + misra + tidy (advisory).
#
# The sub-Makefiles all build on a single `$(CC) ... $(WARN) ...` line, so the
# sanitizer is injected purely by overriding CC/WARN -- no per-suite edits.
# =============================================================================
SAN_CC   ?= g++ -fsanitize=address,undefined -fno-sanitize-recover=all -O1
SAN_WARN ?= -Wall -Wextra -Wpedantic -g
SAN_ENV  ?= UBSAN_OPTIONS=print_stacktrace=1 ASAN_OPTIONS=detect_leaks=0

# A representative hosted+quant corner for the standalone analyzers. Templates
# only diagnose what a translation unit instantiates, so enable the broad gates.
ANALYZE_INC = -I cpp -I cpp/include -I include
ANALYZE_DEF = -DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1 \
              -DTINYMIND_ENABLE_QUANTIZATION=1 -DTINYMIND_ENABLE_HOSTED_IO=1 \
              -DTINYMIND_ENABLE_OSTREAMS=1 -DTINYMIND_ENABLE_FP16=1
# cpp/ is header-only (one .cpp), so a directory scan only analyzes
# lookupTables.cpp and reaches the templates as mere includes. List the headers
# explicitly so cppcheck analyzes each as a translation unit.
ANALYZE_SRCS = $(shell find cpp -name '*.hpp') cpp/lookupTables.cpp

# The QValue suites run under the sanitizers but stay out of COV_SUITES: the
# oracle instantiates ~130 QValue formats, which would swamp the coverage
# gate's function count with one-off template instances. They are the suites
# that catch QValue undefined behaviour (signed overflow in WrapPolicy and in
# fromParts was found this way), so they must run under UBSan in CI.
SAN_EXTRA_SUITES = unit_test/qformat unit_test/qformat_oracle

sanitize :
	@for d in $(COV_SUITES) $(COV_EXAMPLES) $(SAN_EXTRA_SUITES); do \
		echo "=== sanitize: $$d ==="; \
		( cd $$d && $(MAKE) clean >/dev/null 2>&1 && \
		  $(MAKE) CC="$(SAN_CC)" WARN="$(SAN_WARN)" >/dev/null && \
		  $(SAN_ENV) $(MAKE) CC="$(SAN_CC)" WARN="$(SAN_WARN)" run ) \
		  || { echo "SANITIZE FAIL: $$d"; exit 1; }; \
	done
	@echo "sanitize: ASan+UBSan clean across all runtime suites and int8 examples"

# Soak run of the QValue oracle under ASan+UBSan: a fresh random sample
# (ORACLE_SEED) at ORACLE_SCALE times the per-PR random counts. The nightly
# soak workflow runs this with a new seed each night; a failure reproduces
# exactly with the seed and scale it prints. ORACLE_SEED=0 is the per-PR
# sample. Not a PR gate.
ORACLE_SEED  ?= 1
ORACLE_SCALE ?= 100

oracle-soak :
	cd unit_test/qformat_oracle && $(MAKE) clean >/dev/null 2>&1 && \
	  $(MAKE) CC="$(SAN_CC)" WARN="$(SAN_WARN)" >/dev/null && \
	  $(SAN_ENV) TINYMIND_ORACLE_SEED=$(ORACLE_SEED) TINYMIND_ORACLE_SCALE=$(ORACLE_SCALE) ./output/qformat_oracle_unit_test

# ASan+UBSan with the AVX2 SIMD backend active. The plain sanitize target runs
# scalar-only, so the hand-written intrinsics in cpp/include/simd/ otherwise
# never execute under a sanitizer: vector loads that overread a ragged tail or
# walk past a buffer edge are invisible to the scalar build. The suite list is
# the dispatch consumers (QDense / QConv2D / depthwise / pointwise), i.e. the
# quantization tests plus the conv-heavy int8 examples whose golden-output
# checks double as a layer-level scalar-vs-AVX2 equivalence assertion (the
# backend contract is bit-exact, so PASS thresholds tuned on scalar must hold).
# x86-only; pair with `fuzz/fuzz_simd_avx2_diff` for the primitive-level
# differential. Reuses TSAN_SUITES -- same dispatch-consumer list.
SAN_AVX2_CC ?= g++ -fsanitize=address,undefined -fno-sanitize-recover=all -O1 \
               -mavx2 -DTINYMIND_ENABLE_SIMD_AVX2=1

sanitize-avx2 :
	@for d in $(TSAN_SUITES); do \
		echo "=== sanitize-avx2: $$d ==="; \
		( cd $$d && $(MAKE) clean >/dev/null 2>&1 && \
		  $(MAKE) CC="$(SAN_AVX2_CC)" WARN="$(SAN_WARN)" >/dev/null && \
		  $(SAN_ENV) $(MAKE) CC="$(SAN_AVX2_CC)" WARN="$(SAN_WARN)" run ) \
		  || { echo "SANITIZE-AVX2 FAIL: $$d"; exit 1; }; \
	done
	@echo "sanitize-avx2: ASan+UBSan clean with the AVX2 backend active"

# TSan over the OpenMP conv path. TINYMIND_ENABLE_OPENMP=1 parallelizes the
# QConv2D / QConv2DPerChannel output-filter loop -- the only concurrent code in
# the library -- and no other gate can see a data race there: ASan/UBSan don't
# detect races, the static analyzers can't prove thread-safety of the
# parallel-for, and a racy test can pass 999 runs out of 1000. The suite list
# is the quantization tests plus the conv-heavy int8 examples, i.e. everything
# that instantiates the parallelized loop. Clang + libomp is required: GCC's
# libgomp is not TSan-annotated and reports false races on its own barriers
# (sudo apt install clang libomp-dev). OMP_NUM_THREADS is pinned > 1 so the
# loop actually runs concurrently on small CI runners.
TSAN_CC     ?= clang++ -fsanitize=thread -fopenmp -DTINYMIND_ENABLE_OPENMP=1 -O1
TSAN_WARN   ?= -Wall -Wextra -Wpedantic -g
# ignore_noninstrumented_modules + the called_from_lib suppression silence the
# false races TSan reports inside the (uninstrumented) packaged libomp runtime;
# races between tinymind's own instrumented loop iterations still report.
#
# Archer (libarcher.so, ships with libomp-dev) must be loaded via
# OMP_TOOL_LIBRARIES: the packaged libomp's internal synchronization is
# invisible to TSan, so without Archer the implicit barrier at the end of a
# parallel-for carries no happens-before edge and every post-loop read of the
# output buffer reports as a false race against the loop's writes. Archer
# restores those edges through OMPT callbacks.
TSAN_ARCHER ?= $(firstword $(wildcard /usr/lib/llvm-*/lib/libarcher.so))
TSAN_ENV    ?= TSAN_OPTIONS="halt_on_error=1 ignore_noninstrumented_modules=1 suppressions=$(CURDIR)/.tsan-suppressions" \
               OMP_TOOL_LIBRARIES=$(TSAN_ARCHER) OMP_NUM_THREADS=4
TSAN_SUITES = unit_test/quantization examples/resnet_block_int8 \
              examples/resnet18_block_int8 examples/mobilenetv2_int8 \
              examples/kws_cortex_m_int8

tsan :
	@for d in $(TSAN_SUITES); do \
		echo "=== tsan: $$d ==="; \
		( cd $$d && $(MAKE) clean >/dev/null 2>&1 && \
		  $(MAKE) CC="$(TSAN_CC)" WARN="$(TSAN_WARN)" >/dev/null && \
		  $(TSAN_ENV) $(MAKE) CC="$(TSAN_CC)" WARN="$(TSAN_WARN)" run ) \
		  || { echo "TSAN FAIL: $$d"; exit 1; }; \
	done
	@echo "tsan: no data races in the OpenMP conv path"

cppcheck :
	@command -v cppcheck >/dev/null 2>&1 || { echo "ERROR: cppcheck not found. sudo apt install cppcheck"; exit 1; }
	cppcheck --enable=warning,portability --std=c++17 --language=c++ \
	         --inline-suppr --error-exitcode=2 --quiet \
	         --suppressions-list=.cppcheck-suppressions \
	         --suppress=missingIncludeSystem --suppress=missingInclude \
	         $(ANALYZE_INC) $(ANALYZE_DEF) $(ANALYZE_SRCS)
	@echo "cppcheck: clean (all headers analyzed)"

# MISRA C:2012 addon (advisory). Pass MISRA_RULE_TEXTS=<file> to expand rule IDs
# into full descriptions using the licensed MISRA rule text; without it cppcheck
# reports rule IDs only. Never fails the build -- this is the MISRA C ruleset on
# C++ source, so findings are expected and for review, not a pass/fail gate.
MISRA_RULE_TEXTS ?=
MISRA_TEXTS_ARG = $(if $(MISRA_RULE_TEXTS),--rule-texts=$(MISRA_RULE_TEXTS),)
misra :
	@command -v cppcheck >/dev/null 2>&1 || { echo "ERROR: cppcheck not found. sudo apt install cppcheck"; exit 1; }
	-cppcheck --addon=misra $(MISRA_TEXTS_ARG) --enable=style --std=c++17 --language=c++ \
	          --inline-suppr --quiet \
	          --suppress=missingIncludeSystem --suppress=missingInclude \
	          $(ANALYZE_INC) $(ANALYZE_DEF) $(ANALYZE_SRCS) 2>misra-report.txt
	@echo "=== MISRA C:2012 findings by rule (top 25) ==="
	@grep -oE "misra-c2012-[0-9.]+" misra-report.txt 2>/dev/null | sort | uniq -c | sort -rn | head -25 || true
	@echo "total MISRA findings: $$(grep -c 'misra-c2012-' misra-report.txt 2>/dev/null || echo 0)  (full list in misra-report.txt)"
	@echo "misra: advisory MISRA C:2012 report (not a gate; MISRA C++ needs a qualified commercial tool)"

# Bear captures the exact per-TU flags (each suite uses different -D corners),
# which clang-tidy needs to instantiate the templates it diagnoses. `make check`
# is the build that touches every translation unit.
compile_commands.json :
	@command -v bear >/dev/null 2>&1 || { echo "ERROR: bear not found. sudo apt install bear"; exit 1; }
	bear --output compile_commands.json -- $(MAKE) check
	@# Drop the builds that are meant to fail to compile (EXPECT_FAIL in
	@# unit_test/embedded/Makefile), or clang-tidy reports them as library errors.
	python3 -I tools/filter_compile_commands.py compile_commands.json

tidy : compile_commands.json
	@command -v run-clang-tidy >/dev/null 2>&1 || { echo "ERROR: run-clang-tidy not found. sudo apt install clang-tidy"; exit 1; }
	-run-clang-tidy -p . -quiet -header-filter='.*/cpp/.*\.hpp$$' '$(CURDIR)/(cpp|unit_test)/.*' 2>/dev/null | tee tidy-report.txt
	@echo "tidy: advisory report written to tidy-report.txt (not a gate)"

# Header self-containment (HARD GATE). Header-only library: each cpp/**/*.hpp
# must compile as its own translation unit. The library's include contract is
# "tinymind_platform.hpp first, then the header" (see CLAUDE.md), so that prelude
# is injected before each header. Bundled builds mask missing includes because
# an earlier TU drags in <cstddef>/<cstdint> et al.; this catches them one file
# at a time. Fails if any header is not self-contained -- fix by adding the
# missing standard include to the header itself.
PLATFORM_HDR = cpp/include/tinymind_platform.hpp
# HDR_STD / HDR_FLAGS let check-cxx14 rerun the same pass at the language floor.
HDR_STD   ?= c++17
HDR_FLAGS ?=
.PHONY : header-selfcheck
header-selfcheck :
	@pass=0; fail=0; failed=""; \
	for h in $$(find cpp -name '*.hpp' | sort); do \
	  if g++ -std=$(HDR_STD) $(HDR_FLAGS) -fsyntax-only $(ANALYZE_INC) $(ANALYZE_DEF) \
	         -include $(PLATFORM_HDR) -include "$$h" -xc++ /dev/null 2>/dev/null; then \
	    pass=$$((pass+1)); \
	  else \
	    fail=$$((fail+1)); failed="$$failed $$h"; \
	  fi; \
	done; \
	echo "header-selfcheck (-std=$(HDR_STD)): $$pass self-contained, $$fail not"; \
	if [ -n "$$failed" ]; then \
	  echo "Does not compile standalone at -std=$(HDR_STD) $(HDR_FLAGS) (a missing standard include, or under check-cxx14 a construct newer than C++14):"; \
	  for f in $$failed; do echo "  $$f"; done; \
	  exit 1; \
	fi

# Conversion / promotion warnings (ADVISORY). -Wconversion + -Wsign-conversion
# are the highest-signal warnings for fixed-point Q-format code (implicit narrow
# / sign flips in QValue shifts and casts); -Wdouble-promotion catches an
# accidental float->double that silently pulls software double math onto a
# single-precision FPU. Not in the -Werror baseline because template
# instantiations produce expected fixed-point narrowings; this reports the count
# so it can be reviewed and ratcheted. Compiled over the broad hosted+quant
# corner of the embedded smoke source.
STRICT_WARN = -Wconversion -Wsign-conversion -Wdouble-promotion -Wshadow
STRICT_SRC  = unit_test/embedded/embedded_smoke_test.cpp
.PHONY : warnings-strict
warnings-strict :
	@g++ -std=c++17 -fsyntax-only $(ANALYZE_INC) $(ANALYZE_DEF) $(STRICT_WARN) \
	    $(STRICT_SRC) 2>strict-warnings.txt || true
	@n=$$(grep -c 'warning:' strict-warnings.txt 2>/dev/null || echo 0); \
	echo "=== conversion/promotion warnings by kind ==="; \
	grep -oE '\[-W[a-z-]+\]' strict-warnings.txt 2>/dev/null | sort | uniq -c | sort -rn || true; \
	echo "warnings-strict: $$n advisory finding(s) (full list in strict-warnings.txt)"

# Clang conformance build. tinymind is header-only: users compile these headers
# with their own toolchain, so a GCC-only CI lets non-conforming constructs
# reach them unflagged. Clang earns its slot on two behaviours GCC does not
# reproduce -- it diagnoses template member bodies eagerly rather than deferring
# to instantiation, and it owns -W names GCC lacks (and vice versa, which is
# what TINYMIND_DISABLE_WARNING_GCC_ONLY / _CLANG_ONLY in include/compiler.h
# exist to express). Sub-Makefiles assign CC=g++, and a command-line override
# beats a makefile assignment and propagates through MAKEFLAGS, so passing
# CC here is enough -- no sub-Makefile edits needed.
#
# unit_test/integration is absent from CLANG_SUITES: it is a golden-byte suite
# that shells out to already-built example binaries (examples/*/output/<name> --golden)
# rather than compiling anything itself, so running it here only reports
# whether `make check` happened to leave those artifacts behind -- exit 127 on
# a clean tree. Covering the exemplars under clang means building the examples
# with clang first, which is a larger change than this target is scoped for.
CLANG_CXX    ?= clang++
CLANG_SUITES  = unit_test/nn unit_test/qformat unit_test/qformat_oracle unit_test/qlearn \
                unit_test/lookuptable unit_test/quantization unit_test/dual \
                unit_test/kan unit_test/pinn unit_test/ltc unit_test/cfc

check-clang :
	@command -v $(CLANG_CXX) >/dev/null 2>&1 || \
	  { echo "ERROR: $(CLANG_CXX) not found. sudo apt install clang"; exit 1; }
	@echo "=== check-clang: unit_test/embedded (all gate corners) ==="
	@( cd unit_test/embedded && $(MAKE) clean >/dev/null 2>&1 && \
	   $(MAKE) CC="$(CLANG_CXX)" && $(MAKE) CC="$(CLANG_CXX)" run && \
	   $(MAKE) CC="$(CLANG_CXX)" qtree_static_regressions ) \
	   || { echo "CHECK-CLANG FAIL: unit_test/embedded"; exit 1; }
	@for d in $(CLANG_SUITES); do \
		echo "=== check-clang: $$d ==="; \
		( cd $$d && $(MAKE) clean >/dev/null 2>&1 && \
		  $(MAKE) CC="$(CLANG_CXX)" && $(MAKE) CC="$(CLANG_CXX)" run ) \
		  || { echo "CHECK-CLANG FAIL: $$d"; exit 1; }; \
	done
	@echo "check-clang: every suite builds and passes with $(CLANG_CXX)"

# Same bar with the AVX2 backend compiled in and executing. The scalar
# check-clang never reaches the hand-written intrinsics in cpp/include/simd/,
# and intrinsic availability + target-attribute handling is exactly where the
# two compilers diverge. All GitHub-hosted x64 runners have AVX2.
CLANG_AVX2_CC ?= $(CLANG_CXX) -mavx2 -DTINYMIND_ENABLE_SIMD_AVX2=1

check-clang-avx2 :
	@command -v $(CLANG_CXX) >/dev/null 2>&1 || \
	  { echo "ERROR: $(CLANG_CXX) not found. sudo apt install clang"; exit 1; }
	@for d in $(TSAN_SUITES); do \
		echo "=== check-clang-avx2: $$d ==="; \
		( cd $$d && $(MAKE) clean >/dev/null 2>&1 && \
		  $(MAKE) CC="$(CLANG_AVX2_CC)" && $(MAKE) CC="$(CLANG_AVX2_CC)" run ) \
		  || { echo "CHECK-CLANG-AVX2 FAIL: $$d"; exit 1; }; \
	done
	@echo "check-clang-avx2: AVX2 intrinsics build and pass with $(CLANG_CXX)"

# The examples unit_test/integration shells out to. That suite compiles nothing
# itself -- it runs `examples/*/output/<name> --golden` and compares bytes -- so
# covering it under clang means building its inputs with clang first. Worth
# doing: the int8 pipelines are integer-only and therefore ought to be bit-exact
# across compilers, and this target is what actually proves it. The committed
# goldens are the same either way, so a divergence here is a real finding.
CLANG_INTEGRATION_EXAMPLES = anfis_mackey_glass anfis_mackey_glass_int8 \
                             gbdt_tabular_int8 mixed_precision_kws \
                             mixed_precision_mlp_int8_qformat mobilenetv2_int8 \
                             resnet18_block_int8 seq2seq_int8 \
                             seq2seq_softmax_int8 state_space_int8 \
                             tiny_generate_int8 transformer_encoder_int8

check-clang-integration :
	@command -v $(CLANG_CXX) >/dev/null 2>&1 || \
	  { echo "ERROR: $(CLANG_CXX) not found. sudo apt install clang"; exit 1; }
	@for e in $(CLANG_INTEGRATION_EXAMPLES); do \
		echo "=== check-clang-integration: building examples/$$e ==="; \
		( cd examples/$$e && $(MAKE) clean >/dev/null 2>&1 && \
		  $(MAKE) CC="$(CLANG_CXX)" >/dev/null ) \
		  || { echo "CHECK-CLANG-INTEGRATION BUILD FAIL: examples/$$e"; exit 1; }; \
	done
	@echo "=== check-clang-integration: unit_test/integration ==="
	@( cd unit_test/integration && $(MAKE) clean >/dev/null 2>&1 && \
	   $(MAKE) CC="$(CLANG_CXX)" && $(MAKE) CC="$(CLANG_CXX)" run ) \
	   || { echo "CHECK-CLANG-INTEGRATION FAIL: unit_test/integration"; exit 1; }
	@echo "check-clang-integration: golden bytes match with $(CLANG_CXX)"

# Every example under clang, discovered rather than listed so a new example is
# covered the day it is added instead of the day someone remembers to add it
# here. This is a regression guard, not a bug finder: examples/maze and
# examples/dqn_maze had been failing to build under clang unnoticed (an unused
# parameter under -Werror, the same class as the library-header issue in #168)
# precisely because the clang gates covered the unit-test suites and the twelve
# integration exemplars but nothing else.
#
# Builds each example's default target, then its `release` target where one
# exists -- examples/perf_matrix has no `release`, it builds per-ISA variants
# instead, so the interface is probed rather than assumed. Both levels are
# built because -O3 can surface diagnostics a debug build does not.
#
# No -maxdepth on the find: not every example sits one level down.
# examples/pytorch/xor and examples/pytorch_quant/xor are nested a level
# deeper, and capping the depth silently skipped both -- the same
# under-coverage this target exists to prevent.
check-clang-examples :
	@command -v $(CLANG_CXX) >/dev/null 2>&1 || \
	  { echo "ERROR: $(CLANG_CXX) not found. sudo apt install clang"; exit 1; }
	@n=0; \
	for m in $$(find examples -mindepth 2 -name Makefile | sort); do \
		d=$$(dirname $$m); \
		echo "=== check-clang-examples: $$d ==="; \
		( cd $$d && $(MAKE) clean >/dev/null 2>&1 && \
		  $(MAKE) CC="$(CLANG_CXX)" >/dev/null ) \
		  || { echo "CHECK-CLANG-EXAMPLES FAIL (default): $$d"; exit 1; }; \
		if grep -qE '^release[ 	]*:' $$m; then \
			( cd $$d && $(MAKE) CC="$(CLANG_CXX)" release >/dev/null ) \
			  || { echo "CHECK-CLANG-EXAMPLES FAIL (release): $$d"; exit 1; }; \
		fi; \
		n=$$((n+1)); \
	done; \
	echo "check-clang-examples: $$n examples build with $(CLANG_CXX)"

# C++14 language floor (HARD GATE). The README promises C++14, but every other
# build compiles at the compiler's default (C++17 on current GCC and Clang), so
# nothing else notices a C++17-only construct in cpp/. Three layers:
#
# 1. Every cpp/**/*.hpp standalone at -std=c++14 -pedantic-errors. Without
#    -pedantic-errors GCC accepts if constexpr, inline variables and fold
#    expressions at C++14 with only a warning.
# 2. The embedded gate corners and the CXX14_SUITES Boost suites, built and
#    run. Linking is what catches a static constexpr member read at run time
#    with no pre-C++17 out-of-class definition; a syntax-only pass cannot, and
#    FFT1D / QFFT1D had exactly that defect.
# 3. Every example, built (discovered from the tree, as in check-clang-examples).
#
# apps/ are host tools, not the library, and are exempt: apps/activation needs
# C++17 (std::filesystem) and says so in its own Makefile.
CXX14_CC     ?= g++ -std=c++14
CXX14_SUITES  = $(CLANG_SUITES)

check-cxx14 :
	@$(MAKE) --no-print-directory header-selfcheck HDR_STD=c++14 HDR_FLAGS=-pedantic-errors
	@echo "=== check-cxx14: unit_test/embedded (all gate corners) ==="
	@( cd unit_test/embedded && $(MAKE) clean >/dev/null 2>&1 && \
	   $(MAKE) CC="$(CXX14_CC)" >/dev/null && $(MAKE) CC="$(CXX14_CC)" run >/dev/null ) \
	   || { echo "CHECK-CXX14 FAIL: unit_test/embedded"; exit 1; }
	@for d in $(CXX14_SUITES); do \
		echo "=== check-cxx14: $$d ==="; \
		( cd $$d && $(MAKE) clean >/dev/null 2>&1 && \
		  $(MAKE) CC="$(CXX14_CC)" >/dev/null && $(MAKE) CC="$(CXX14_CC)" run >/dev/null ) \
		  || { echo "CHECK-CXX14 FAIL: $$d"; exit 1; }; \
	done
	@n=0; \
	for m in $$(find examples -mindepth 2 -name Makefile | sort); do \
		d=$$(dirname $$m); \
		( cd $$d && $(MAKE) clean >/dev/null 2>&1 && $(MAKE) CC="$(CXX14_CC)" >/dev/null ) \
		  || { echo "CHECK-CXX14 FAIL (example): $$d"; exit 1; }; \
		n=$$((n+1)); \
	done; \
	echo "check-cxx14: headers, embedded corners, Boost suites and $$n examples build (and the suites pass) at $(CXX14_CC)"

# MemorySanitizer over the embedded smoke corners.
#
# MSan closes a gap ASan and UBSan structurally cannot: reads of uninitialized
# memory. Both UB sites fixed in #171 -- a static-initialization-order read and
# an uninitialized member -- were invisible to the existing sanitizer jobs, and
# that is why they survived so long.
#
# Two deliberate constraints:
#
# 1. Embedded smoke only, not the Boost suites. MSan demands that *every* linked
#    object be instrumented; against a stock libstdc++ the first report lands
#    inside Boost.Test's own static init and the run dies before reaching any
#    tinymind code. Covering the Boost suites means building libc++ and
#    Boost.Test with MSan, which is a much larger undertaking. The embedded
#    smoke test uses neither, and it is the code that actually ships.
#
# 2. -O0, not -O1. At -O1 and above the optimizer folds an undef read away
#    before instrumentation sees it: a deliberately planted uninitialized read
#    is reported at -O0 and silently missed at -O1. An MSan job at -O1 would be
#    vacuous, so the level here is load-bearing, not a debug convenience.
MSAN_CC ?= $(CLANG_CXX) -fsanitize=memory -fsanitize-memory-track-origins=2 \
           -fno-omit-frame-pointer -O0 -g

check-msan :
	@command -v $(CLANG_CXX) >/dev/null 2>&1 || \
	  { echo "ERROR: $(CLANG_CXX) not found. sudo apt install clang"; exit 1; }
	@( cd unit_test/embedded && $(MAKE) clean >/dev/null 2>&1 && \
	   $(MAKE) CC="$(MSAN_CC)" WARN="$(SAN_WARN)" && \
	   $(MAKE) CC="$(MSAN_CC)" WARN="$(SAN_WARN)" run ) \
	   || { echo "CHECK-MSAN FAIL: unit_test/embedded"; exit 1; }
	@$(MAKE) -C fuzz msan-replay FUZZ_CXX=$(CLANG_CXX) \
	   || { echo "CHECK-MSAN FAIL: fuzz corpus"; exit 1; }
	@echo "check-msan: no uninitialized reads across the embedded gate corners or the fuzz corpus"

# The same sanitizer over the Boost suites, which check-msan cannot reach.
#
# check-msan is confined to unit_test/embedded because MSan requires every
# linked object to be instrumented: against a stock libstdc++ the Boost suites
# abort inside Boost.Test's own static initialization, and those reports are
# artifacts of the uninstrumented runtime rather than findings. That is a real
# gap, since the static-initialization-order UB fixed in #171 lived in exactly
# such a suite.
#
# This target closes it by linking against a libc++ built with MSan. Building
# that runtime takes 10-20 minutes, which is why this is driven by the nightly
# .github/workflows/msan-nightly.yml rather than a PR gate; the suites
# themselves run in seconds once it exists. Point MSAN_LIBCXX_PREFIX at an
# install tree produced with -DLLVM_USE_SANITIZER=MemoryWithOrigins.
#
# Boost.Test needs no separate treatment: the suites include the header-only
# <boost/test/included/unit_test.hpp>, so it is compiled into each translation
# unit and instrumented with it. The C++ runtime was the only uninstrumented
# piece.
#
# -O0 for the same reason as check-msan: at -O1 and above the optimizer folds
# an undefined read away before instrumentation observes it, which would leave
# the job green and blind.
#
# What this target treats as failure needs stating, because it is not simply
# "the suite exited non-zero". Swapping the C++ runtime changes the random
# numbers: std::uniform_real_distribution is not required to produce the same
# sequence across implementations, and libc++ and libstdc++ genuinely differ.
# Training tests seeded from it therefore start from different weights and land
# outside tolerances tuned on libstdc++ -- observed in
# test_case_lstm_weight_serialization (0.0236 against a 0.02 bound) and
# test_case_rmsprop_fixedpoint_xor (average error 25 against a bound of 4).
# That is a property of the runtime swap, not a defect, and failing on it would
# make this job permanently red for a reason it was never asking about.
#
# So the criterion is narrow and explicit:
#   - any MemorySanitizer report            -> FAIL (this is the whole point)
#   - Boost "failures are detected" in output -> noted, not fatal (runtime RNG)
#   - any other non-zero exit               -> FAIL (crash, signal, build error)
# A genuine crash is still caught; only numeric divergence is tolerated.
#
# Boost assertion failures are recognized from the log rather than from the
# exit status. Boost.Test exits 201, but this runs it through a sub-make, and
# make reports its own failure as 2 -- so matching on 201 silently never fires
# and every divergence looked like a crash.
MSAN_LIBCXX_PREFIX ?=
MSAN_LIBCXX_CC      = $(CLANG_CXX) -stdlib=libc++ -nostdinc++ \
                      -isystem $(MSAN_LIBCXX_PREFIX)/include/c++/v1 \
                      -L$(MSAN_LIBCXX_PREFIX)/lib \
                      -Wl,-rpath,$(MSAN_LIBCXX_PREFIX)/lib \
                      -fsanitize=memory -fsanitize-memory-track-origins=2 \
                      -fno-omit-frame-pointer -O0 -g
MSAN_LIBCXX_SUITES  = unit_test/nn unit_test/qformat unit_test/qformat_oracle unit_test/qlearn \
                      unit_test/lookuptable unit_test/quantization \
                      unit_test/dual unit_test/kan unit_test/pinn \
                      unit_test/ltc unit_test/cfc

check-msan-libcxx :
	@test -n "$(MSAN_LIBCXX_PREFIX)" || \
	  { echo "ERROR: set MSAN_LIBCXX_PREFIX to an MSan-instrumented libc++ install tree."; exit 1; }
	@test -d "$(MSAN_LIBCXX_PREFIX)/include/c++/v1" || \
	  { echo "ERROR: $(MSAN_LIBCXX_PREFIX)/include/c++/v1 not found -- not a libc++ install tree."; exit 1; }
	@rm -f msan-libcxx.log
	@fail=0; diverged=""; \
	for d in $(MSAN_LIBCXX_SUITES); do \
		echo "=== check-msan-libcxx: $$d ==="; \
		( cd $$d && $(MAKE) clean >/dev/null 2>&1 && \
		  $(MAKE) CC="$(MSAN_LIBCXX_CC)" WARN="$(SAN_WARN)" >/dev/null ) \
		  || { echo "CHECK-MSAN-LIBCXX BUILD FAIL: $$d"; exit 1; }; \
		( cd $$d && $(MAKE) CC="$(MSAN_LIBCXX_CC)" WARN="$(SAN_WARN)" run ) \
		  >$$(basename $$d).msan.log 2>&1; rc=$$?; \
		cat $$(basename $$d).msan.log; \
		cat $$(basename $$d).msan.log >> msan-libcxx.log; \
		if grep -q "MemorySanitizer" $$(basename $$d).msan.log; then \
			echo "CHECK-MSAN-LIBCXX FAIL: $$d reported an uninitialized read"; fail=1; \
		elif grep -q "failures are detected" $$(basename $$d).msan.log; then \
			echo "note: $$d has Boost assertion failures but no MSan report (see libc++ RNG note)"; \
			diverged="$$diverged $$d"; \
		elif [ $$rc -ne 0 ]; then \
			echo "CHECK-MSAN-LIBCXX FAIL: $$d exited $$rc (crash or signal, not a test assertion)"; fail=1; \
		fi; \
		rm -f $$(basename $$d).msan.log; \
	done; \
	if [ -n "$$diverged" ]; then \
		echo "check-msan-libcxx: numeric divergence under libc++ in:$$diverged"; \
	fi; \
	test $$fail -eq 0 || exit 1; \
	echo "check-msan-libcxx: no uninitialized reads across the Boost suites"

analyze : misra tidy cppcheck header-selfcheck warnings-strict

.PHONY : sanitize cppcheck misra tidy analyze coverage-check header-selfcheck warnings-strict check-cxx14 \
         check-clang check-clang-avx2 check-clang-integration check-clang-examples \
         check-msan check-msan-libcxx oracle-soak

# Recursively clean every unit test, example, and app (each subdir Makefile has
# its own clean target), plus the coverage artifacts.
clean : coverage-clean
	rm -rf compile_commands.json tidy-report.txt strict-warnings.txt misra-report.txt
	@for m in $$(find unit_test examples apps -name Makefile); do \
		d=$$(dirname $$m); \
		echo "clean $$d"; \
		$(MAKE) -C $$d clean >/dev/null 2>&1 || true; \
	done