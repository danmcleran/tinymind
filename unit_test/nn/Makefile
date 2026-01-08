# Simple Makefile for the neural network unit tests
# Enable C++ iostreams
# Tell the code to compile the activation function LUTs used by the unit tests
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
SOURCES=nn_unit_test.cpp ../../cpp/lookupTables.cpp
DEFINES=-DTINYMIND_ENABLE_OSTREAMS=1 -DTINYMIND_USE_SIGMOID_8_8=1 -DTINYMIND_USE_SIGMOID_16_16=1 -DTINYMIND_USE_LOG_16_16=1 -DTINYMIND_USE_TANH_8_8=1 -DTINYMIND_USE_TANH_8_24=1 -DTINYMIND_USE_TANH_16_16=1 -DTINYMIND_USE_EXP_16_16=1 -DTINYMIND_USE_EXP_8_8=1
INCLUDES=-I ../../include -I../../cpp -I../../cpp/include -I../../include -I${BOOST_HOME}
OUT=./output/nn_unit_test
nn_unit_test:
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(DEFINES) $(INCLUDES)
run :
	cd ./output && ./nn_unit_test && cd ../

clean:
	rm -f ./output/*
