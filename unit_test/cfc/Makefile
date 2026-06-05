# Makefile for the Closed-form Continuous-time cell (cpp/cfc.hpp) unit tests.
#
# FLOAT + STD enabled: the double path uses std::exp/tanh and float
# divide; reverse-mode training (revdual.hpp) and the RevVar Constant<> lift are
# gated on FLOAT && STD. lookupTables.cpp + the SIGMOID/TANH Q16.16 table macros
# back the QValue activation overloads so the fixed-point inference path that
# cfc.hpp also supports stays linkable (regression-guarded by the example smoke
# build and unit_test/embedded for the freestanding corner).
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
SOURCES=cfc_unit_test.cpp ../../cpp/lookupTables.cpp
DEFINES=-DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1 \
        -DTINYMIND_USE_TANH_16_16=1 -DTINYMIND_USE_SIGMOID_16_16=1
INCLUDES=-I../../include -I../../cpp -I../../cpp/include -I${BOOST_HOME}
OUT=./output/cfc_unit_test

cfc_unit_test:
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(DEFINES) $(INCLUDES)

run:
	cd ./output && ./cfc_unit_test && cd ../

coverage:
	$(MKDIR)
	$(CC) --coverage -O0 -g $(WARN) -o $(OUT) $(SOURCES) $(DEFINES) $(INCLUDES)

clean:
	rm -f ./output/*
