# Simple Makefile for the XOR neural network example
# Enable C++ iostreams
# Tell the code to compile the Q8.8 tanh activation function LUT
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/xor
SOURCES=xor.cpp ../../../cpp/lookupTables.cpp
INCLUDES=-I../../../cpp -I../../../cpp/include -I../../../include/
DEFINES=-DTINYMIND_ENABLE_OSTREAMS=1 -DTINYMIND_USE_SIGMOID_16_16=1

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -ggdb -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

run :
	cd ./output && ./xor && cd ../

plot :
	python3 ../../../unit_test/nn/nn_plot.py output/nn_fixed_xor.txt

clean :
	rm -f ./output/*
