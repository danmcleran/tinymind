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
		     --output-file $$out --rc geninfo_unexecuted_blocks=1 $(LCOV_IGNORE) || true; \
		[ -s $$out ] && echo "-a $$out" >> coverage/merge_args; \
	done
	lcov $$(cat coverage/merge_args) --output-file coverage/all.info $(LCOV_IGNORE)
	lcov --extract coverage/all.info '*/cpp/*' \
	     --output-file coverage/tinymind.info $(LCOV_IGNORE)
	genhtml coverage/tinymind.info --output-directory coverage/html $(GENHTML_IGNORE)
	python3 tools/coverage_dashboard.py coverage/tinymind.info coverage/dashboard.html
	@echo "Dashboard:   coverage/dashboard.html"
	@echo "HTML report: coverage/html/index.html"

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

# Recursively clean every unit test, example, and app (each subdir Makefile has
# its own clean target), plus the coverage artifacts.
clean : coverage-clean
	@for m in $$(find unit_test examples apps -name Makefile); do \
		d=$$(dirname $$m); \
		echo "clean $$d"; \
		$(MAKE) -C $$d clean >/dev/null 2>&1 || true; \
	done