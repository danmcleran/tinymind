# Simple Makefile for the XOR neural network example
# Enable C++ iostreams
# Tell the code to compile the Q8.8 tanh activation function LUT
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/xor
SOURCES=xor.cpp xornet.cpp ../../cpp/lookupTables.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_OSTREAMS=1 -DTINYMIND_USE_SIGMOID_8_8=1

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)
	g++ -c xornet.cpp $(WARN) -o ./output/xornet.o $(INCLUDES) $(DEFINES)
	g++ -c ../../cpp/lookupTables.cpp $(WARN) -o ./output/lookupTables.o $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)
	g++ -c xornet.cpp -O3 $(WARN) -o ./output/xornet.o $(INCLUDES) $(DEFINES)
	g++ -c ../../cpp/lookupTables.cpp -O3 $(WARN) -o ./output/lookupTables.o $(INCLUDES) $(DEFINES)

run :
	cd ./output && ./xor && cd ../

plot :
	python3 ../../unit_test/nn/nn_plot.py output/nn_fixed_xor.txt

clean :
	rm -f ./output/*
