# Makefile to build and run all unit tests, build examples, and build apps.
default :
	@echo "Use 'make check' to build and run all unit tests."
check :
	cd unit_test/nn && make clean && make && make run && cd -
	cd unit_test/qformat && make clean && make && make run && cd -
	cd unit_test/qlearn && make clean && make && make run && cd -
	cd unit_test/lookuptable && make clean && make && make run && cd -
	cd unit_test/embedded && make clean && make && make run && make simd_prereq_regressions && cd -
	cd unit_test/quantization && make clean && make && make run && cd -
	cd unit_test/dual && make clean && make && make run && cd -
	cd unit_test/pinn && make clean && make && make run && cd -
	cd unit_test/ltc && make clean && make && make run && cd -
	cd unit_test/cfc && make clean && make && make run && cd -
	cd examples/pinn_heat1d && make clean && make && make release && make run && make train && cd -
	cd examples/ltc_sequence && make clean && make && make release && make run && cd -
	cd examples/cfc_sequence && make clean && make && make release && make run && cd -
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
	cd examples/resnet18_block_int8 && make clean && make && make release && make run && cd -
	cd examples/mobilenetv2_int8 && make clean && make && make release && make run && cd -
	cd examples/mixed_precision_kws && make clean && make && make release && make run && cd -
	cd examples/mixed_precision_mlp_int8_qformat && make clean && make && make release && make run && cd -
	cd unit_test/integration && make clean && make && make run && cd -
	cd examples/pytorch_quant/xor && make clean && make && make release && make run && cd -
	cd examples/import_demo && make clean && make && make release && make run && cd -
	cd examples/iris && make clean && make release && make run && cd -
	cd examples/energy_efficiency && make clean && make release && make run && cd -
	cd examples/optical_digits && make clean && make release && make run && cd -
	cd examples/har_activity && make clean && make release && make run && cd -
	cd examples/gas_sensor_drift && make clean && make release && make run && cd -
	cd examples/air_quality && make clean && make release && make run && cd -
	cd examples/lstm_sinusoid_float && make clean && make release && make run && cd -
	cd examples/elman_temporal_xor && make clean && make release && make run && cd -
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

sanitize :
	@for d in $(COV_SUITES) $(COV_EXAMPLES); do \
		echo "=== sanitize: $$d ==="; \
		( cd $$d && $(MAKE) clean >/dev/null 2>&1 && \
		  $(MAKE) CC="$(SAN_CC)" WARN="$(SAN_WARN)" >/dev/null && \
		  $(SAN_ENV) $(MAKE) CC="$(SAN_CC)" WARN="$(SAN_WARN)" run ) \
		  || { echo "SANITIZE FAIL: $$d"; exit 1; }; \
	done
	@echo "sanitize: ASan+UBSan clean across all runtime suites and int8 examples"

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
TSAN_ENV    ?= TSAN_OPTIONS=halt_on_error=1 OMP_NUM_THREADS=4
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

tidy : compile_commands.json
	@command -v run-clang-tidy >/dev/null 2>&1 || { echo "ERROR: run-clang-tidy not found. sudo apt install clang-tidy"; exit 1; }
	-run-clang-tidy -p . -quiet -header-filter='.*/cpp/.*\.hpp$$' '$(CURDIR)/(cpp|unit_test)/.*' 2>/dev/null | tee tidy-report.txt
	@echo "tidy: advisory report written to tidy-report.txt (not a gate)"

analyze : misra tidy cppcheck

.PHONY : sanitize cppcheck misra tidy analyze coverage-check

# Recursively clean every unit test, example, and app (each subdir Makefile has
# its own clean target), plus the coverage artifacts.
clean : coverage-clean
	rm -rf compile_commands.json tidy-report.txt
	@for m in $$(find unit_test examples apps -name Makefile); do \
		d=$$(dirname $$m); \
		echo "clean $$d"; \
		$(MAKE) -C $$d clean >/dev/null 2>&1 || true; \
	done