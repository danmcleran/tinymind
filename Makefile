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
	cd unit_test/integration && make clean && make && make run && cd -
	cd examples/pytorch_quant/xor && make clean && make && make release && make run && cd -
	cd examples/import_demo && make clean && make && make release && make run && cd -
	cd examples/perf_matrix && make clean && make && make report && cd -
	cd apps/activation && make clean && make && make release && cd -