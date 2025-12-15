# Simple Makefile for the neural network unit tests
# Enable C++ iostreams
# Tell the code to compile the activation function LUTs used by the unit tests
nn_unit_test: nn_unit_test.cpp ../../cpp/lookupTables.cpp
#	Make an output dir to hold the executable
	mkdir -p ./output
#	Build the unit tests
	g++ -g -Wall -Wextra -Werror -o ./output/nn_unit_test nn_unit_test.cpp ../../cpp/lookupTables.cpp -DTINYMIND_ENABLE_OSTREAMS=1 -DTINYMIND_USE_SIGMOID_8_8=1 -DTINYMIND_USE_SIGMOID_16_16=1 -DTINYMIND_USE_LOG_16_16=1 -DTINYMIND_USE_TANH_8_8=1 -DTINYMIND_USE_TANH_8_24=1 -DTINYMIND_USE_TANH_16_16=1 -DTINYMIND_USE_EXP_16_16=1 -DTINYMIND_USE_EXP_8_8=1 -I ../../include -I../../cpp -I../../cpp/include -I../../include -I${BOOST_HOME}
clean:
	rm -drf ./output/*
