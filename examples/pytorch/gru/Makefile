# PyTorch GRU -> tinymind parity example.
#
# `make` builds gru_parity, the tinymind side: it loads a gru_export.py weight
# file into GruNeuralNetwork (double or Q16.16) and runs sequences from stdin.
# `make parity` trains the GRU in PyTorch, exports it, and checks that tinymind
# reproduces PyTorch's outputs. It needs a Python with torch and numpy:
#   make parity PYTHON=/path/to/venv/bin/python
# It is not run in CI (torch is not a CI dependency).
MKDIR=mkdir -p ./output
CC=g++
PYTHON?=python3
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/gru_parity
SOURCES=gru_parity.cpp ../../../cpp/lookupTables.cpp
INCLUDES=-I../../../cpp -I../../../cpp/include -I../../../include/
DEFINES=-DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_HOSTED_IO=1 -DTINYMIND_USE_SIGMOID_16_16=1 -DTINYMIND_USE_TANH_16_16=1

default :
	$(MKDIR)
	$(CC) -ggdb $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -ggdb -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

parity : default
	$(PYTHON) test_gru_parity.py --binary $(OUT)

clean :
	rm -f ./output/*
