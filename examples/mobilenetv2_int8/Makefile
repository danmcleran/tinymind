# Phase 16 exemplar: int8 MobileNetV2-shaped pipeline. Mirrors the build
# pattern of examples/resnet18_block_int8 — same CSV bench and golden
# emitters, same Makefile knobs.

MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/mobilenetv2_int8
SOURCES=mobilenetv2_int8.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_QUANTIZATION=1

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

run :
	cd ./output && ./mobilenetv2_int8 && cd ../

coverage :
	$(MKDIR)
	$(CC) --coverage -O0 -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

bench :
	cd ./output && ./mobilenetv2_int8 --bench > mobilenetv2_int8.csv && cd ../

golden :
	cd ./output && ./mobilenetv2_int8 --golden > mobilenetv2_int8.golden && cd ../

clean :
	rm -f ./output/*

plot :
	python3 plot.py
