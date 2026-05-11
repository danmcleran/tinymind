# Phase 16 exemplar: mixed-precision KWS pipeline.
#   int8 CNN feature extractor -> fp16 attention head -> int8 dense classifier.
# Exercises the Phase 9 qbridge converters in production shape.

MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/mixed_precision_kws
SOURCES=mixed_precision_kws.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_QUANTIZATION=1 -DTINYMIND_ENABLE_FP16=1

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

run :
	cd ./output && ./mixed_precision_kws && cd ../

bench :
	cd ./output && ./mixed_precision_kws --bench > mixed_precision_kws.csv && cd ../

golden :
	cd ./output && ./mixed_precision_kws --golden > mixed_precision_kws.golden && cd ../

clean :
	rm -f ./output/*
