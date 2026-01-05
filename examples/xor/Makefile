# Simple Makefile for the XOR neural network example
# Enable C++ iostreams
# Tell the code to compile the Q8.8 tanh activation function LUT
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror
OUT=./output/xor
SOURCES=xor.cpp xornet.cpp ../../cpp/lookupTables.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_OSTREAMS=1 -DTINYMIND_USE_TANH_8_8=1
default :
#	Make an output dir to hold the executable
	$(MKDIR)
#	Build the example with default build flags
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
#	Make an output dir to hold the executable
	$(MKDIR)
#	Build the example with default build flags
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)
run :
#	Run the XOR example
	cd ./output && ./xor && cd ../
plot :
#	Plot the results of the XOR example using matplotlib
	python3 ../../unit_test/nn/nn_plot.py output/nn_fixed_xor.txt

clean :
# Remove all object files
	rm -f ./*.o
	rm -f ./output/*
