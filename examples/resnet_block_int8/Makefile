# Phase 10 demo: ResNet-shaped residual block driven through the
# parallel Q* layer family. Mirrors the build pattern of
# examples/kws_cortex_m_int8.

MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/resnet_block_int8
SOURCES=resnet_block_int8.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_QUANTIZATION=1

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

run :
	cd ./output && ./resnet_block_int8 && cd ../

clean :
	rm -f ./output/*
