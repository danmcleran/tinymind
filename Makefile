# Makefile to build and run all unit tests, build examples, and build apps.
default :
	@echo "Use 'make check' to build and run all unit tests."
check :
	cd unit_test/nn && make clean && make && make run && cd -
	cd unit_test/qformat && make clean && make && make run && cd -
	cd unit_test/qlearn && make clean && make && make run && cd -
	cd examples/xor && make clean && make && cd -
	cd examples/maze && make clean && make && cd -
	cd examples/dqn_maze && make clean && make && cd -
	cd apps/activation && make clean && make && cd -