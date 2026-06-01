# Simple Makefile for the q-learn unit tests
# Enable C++ iostreams
# Tell the code to compile the Q16.16 tanh activation function LUT
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
SOURCES=qlearn_unit_test.cpp ../../cpp/lookupTables.cpp
DEFINES=-DTINYMIND_ENABLE_OSTREAMS=1 -DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_HOSTED_RAND=1
INCLUDES=-I../../cpp -I../../cpp/include -I../../include -DTINYMIND_USE_TANH_16_16=1 -I${BOOST_HOME}
OUT=./output/qlearn_unit_test

qlearn_unit_test:
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(DEFINES) $(INCLUDES)

run:
	cd ./output && ./qlearn_unit_test && cd ../

coverage:
	$(MKDIR)
	$(CC) --coverage -O0 -g $(WARN) -o $(OUT) $(SOURCES) $(DEFINES) $(INCLUDES)

clean:
	rm -f ./output/*
