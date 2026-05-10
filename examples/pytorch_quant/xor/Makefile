# Makefile for the PyTorch -> TinyMind int8 quantization XOR example.
#
# Host build only. TINYMIND_ENABLE_QUANTIZATION turns on the q*.hpp layer
# family; FLOAT + STD let the demo rebuild Requantizers and the qsigmoid
# LUT from the calibration scales emitted by xor_quant.py. The deployable
# MCU shape is FLOAT=0 STD=0 with the integer (multiplier, shift, zero_point)
# triples baked in directly.
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
SOURCES=xor_quant.cpp
INCLUDES=-I../../../cpp -I../../../cpp/include -I../../../include
DEFINES=-DTINYMIND_ENABLE_QUANTIZATION=1 -DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1
OUT=./output/xor_quant

default :
	$(MKDIR)
	$(CC) -ggdb $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -ggdb -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

run :
	cd ./output && ./xor_quant

regenerate-weights :
	python3 xor_quant.py

clean :
	rm -f ./output/*
