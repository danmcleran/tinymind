# KWS-style depthwise-separable CNN example, host runner.
# Demonstrates: Conv2D -> DepthwiseConv2D -> PointwiseConv2D -> GAP pipeline
# with per-layer cycle counts and footprint reporting.

MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/kws_cortex_m
SOURCES=kws_cortex_m.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

run :
	cd ./output && ./kws_cortex_m && cd ../

clean :
	rm -f ./output/*
