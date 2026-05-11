# Phase 15 import_demo Makefile.
#
# Builds an end-to-end PyTorch -> TinyMind int8 importer demonstration that
# does NOT depend on a torch install: a small MLP carries deterministic
# float weights, a 64-sample synthetic dataset drives calibration via the
# Phase 15 observers (Percentile, KL divergence, RangeObserver) and the
# Cross-Layer Equalization pass, then the int8 forward is compared against
# the float reference for parity. FLOAT + STD enabled because calibration
# helpers live behind that gate; QUANT=1 turns on the q*.hpp family.
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
SOURCES=import_demo.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include
DEFINES=-DTINYMIND_ENABLE_QUANTIZATION=1 -DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1
OUT=./output/import_demo

default :
	$(MKDIR)
	$(CC) -ggdb $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -ggdb -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

run :
	cd ./output && ./import_demo

regenerate-pytorch :
	python3 demo.py

clean :
	rm -f ./output/*
