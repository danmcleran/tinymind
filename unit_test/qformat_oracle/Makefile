# Makefile for the QValue differential oracle (see qformat_oracle_unit_test.cpp)
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
SOURCES=qformat_oracle_unit_test.cpp
DEFINES=-DTINYMIND_ENABLE_STD=1
INCLUDES=-I../../include -I../../cpp -I../../cpp/include -I${BOOST_HOME}
OUT=./output/qformat_oracle_unit_test

qformat_oracle_unit_test:
	$(MKDIR)
	$(CC) -g -O1 $(WARN) -o $(OUT) $(SOURCES) $(DEFINES) $(INCLUDES)

run:
	./output/qformat_oracle_unit_test

clean:
	rm -f ./output/*
