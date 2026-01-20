# Simple Makefile for the q-format unit tests
# Enable C++ iostreams
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
SOURCES=qformat_unit_test.cpp
DEFINES=-DTINYMIND_ENABLE_OSTREAMS=1
INCLUDES=-I../../include -I../../cpp -I${BOOST_HOME}
OUT=./output/qformat_unit_test

qformat_unit_test:
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(DEFINES) $(INCLUDES)

run:
	cd ./output && ./qformat_unit_test && cd ../

clean:
	rm -f ./output/*
