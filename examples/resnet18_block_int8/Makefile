# Phase 16 exemplar: int8 ResNet-style stem + 1 stage. Mirrors the build
# pattern of examples/resnet_block_int8 but at larger spatial dimensions.

MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/resnet18_block_int8
SOURCES=resnet18_block_int8.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_QUANTIZATION=1

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

run :
	cd ./output && ./resnet18_block_int8 && cd ../

coverage :
	$(MKDIR)
	$(CC) --coverage -O0 -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

bench :
	cd ./output && ./resnet18_block_int8 --bench > resnet18_block_int8.csv && cd ../

golden :
	cd ./output && ./resnet18_block_int8 --golden > resnet18_block_int8.golden && cd ../

clean :
	rm -f ./output/*

plot :
	python3 plot.py
