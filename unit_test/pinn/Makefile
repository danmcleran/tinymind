# Makefile for the PINN building-block (PinnMlp, sgdStep, sgdStepReverse, heat-eq residual) unit tests.
#
# FLOAT + STD enabled so the double-typed tests and std::tanh in the chain-rule
# check compile in. The Dual<QValue> path itself is freestanding-clean and is
# additionally regression-guarded by unit_test/embedded (freestanding corner).
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
# lookupTables.cpp + the TANH/SIGMOID Q16.16 table macros back the QValue
# activation overloads (tanh(Dual)/sigmoid(Dual)); the double path uses std::.
SOURCES=pinn_unit_test.cpp ../../cpp/lookupTables.cpp
DEFINES=-DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1 \
        -DTINYMIND_USE_TANH_16_16=1 -DTINYMIND_USE_SIGMOID_16_16=1 \
        -DTINYMIND_USE_SIN_16_16=1 -DTINYMIND_USE_COS_16_16=1 -DTINYMIND_USE_EXP_16_16=1
INCLUDES=-I../../include -I../../cpp -I../../cpp/include -I${BOOST_HOME}
OUT=./output/pinn_unit_test

# Built -O2: the heat-equation residual test nests Dual<Dual<MultiDual>> across
# a collocation grid for 3000 epochs -- ~8s at -O0, well under 2s at -O2.
pinn_unit_test:
	$(MKDIR)
	$(CC) -O2 -g $(WARN) -o $(OUT) $(SOURCES) $(DEFINES) $(INCLUDES)

run:
	cd ./output && ./pinn_unit_test && cd ../

coverage:
	$(MKDIR)
	$(CC) --coverage -O0 -g $(WARN) -o $(OUT) $(SOURCES) $(DEFINES) $(INCLUDES)

clean:
	rm -f ./output/*
