# Simple Makefile for the XOR neural network example
# Enable C++ iostreams
# Tell the code to compile the Q8.8 tanh activation function LUT
default :
#	Make an output dir to hold the executable
	mkdir -p ./output
#	Build the example with default build flags
	g++ -O3 -Wall -o ./output/xor xor.cpp xornet.cpp ../../cpp/lookupTables.cpp -DTINYMIND_ENABLE_OSTREAMS=1 -DTINYMIND_USE_TANH_8_8=1 -I../../cpp/ -I../../apps/include

debug  :
#	Make an output dir to hold the executable
	mkdir -p ./output
#	Build the example with default build flags
	g++ -g -Wall -o ./output/xor xor.cpp xornet.cpp ../../cpp/lookupTables.cpp -DTINYMIND_ENABLE_OSTREAMS=1 -DTINYMIND_USE_TANH_8_8=1 -I../../cpp/ -I../../apps/include

# Remove all object files
clean:
	rm -f ./output/*
