# Simple Makefile for the KAN unit tests
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
SOURCES=kan_unit_test.cpp ../../cpp/lookupTables.cpp
DEFINES=-DTINYMIND_ENABLE_OSTREAMS=1 -DTINYMIND_USE_SIGMOID_8_8=1
INCLUDES=-I ../../include -I../../cpp -I../../cpp/include -I../../include -I${BOOST_HOME}
OUT=./output/kan_unit_test
kan_unit_test:
	$(MKDIR)
	$(CC) -ggdb $(WARN) -o $(OUT) $(SOURCES) $(DEFINES) $(INCLUDES)
run :
	./output/kan_unit_test

clean:
	rm -f ./output/*
