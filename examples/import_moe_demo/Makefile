# Phase 3 Mixture-of-Experts importer round-trip.
#
# Builds a host-side PyTorch->TinyMind importer demonstration for the MoE
# layer that does NOT depend on torch: a deterministic float MoE is generated
# by demo.py with numpy, calibrated through apps/import_pytorch, and emitted as
# weights.hpp. This binary consumes weights.hpp + moe_reference.hpp, rebuilds
# cpp/qmoe.hpp::QMixtureOfExperts, and checks parity against the float
# reference. FLOAT + STD enabled (calibration helpers live behind that gate);
# QUANT=1 turns on the q*.hpp family.
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
SOURCES=import_moe_demo.cpp
INCLUDES=-I. -I../../cpp -I../../cpp/include -I../../include
DEFINES=-DTINYMIND_ENABLE_QUANTIZATION=1 -DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1
OUT=./output/import_moe_demo

default :
	$(MKDIR)
	$(CC) -ggdb $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -ggdb -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

run :
	./output/import_moe_demo

regenerate-pytorch :
	python3 demo.py

clean :
	rm -f ./output/*
