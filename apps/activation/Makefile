# Simple Makefile for the activation function LUTs
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
INCLUDES=-I../../cpp/
SOURCES=activationTableGenerator.cpp
OUT=./output/activationTableGenerator

default :
#	Make an output dir to hold the executable
	$(MKDIR)
#	Build the application
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) -DNUMBER_OF_ACTIVATION_TABLE_VALUES=96 -DMIN_X_TABLE_VALUE=-6 -DMAX_X_TABLE_VALUE=6 -DACTIVATION_DELTA_SHIFT=3 -lstdc++fs

release :
#	Make an output dir to hold the executable
	$(MKDIR)
#	Build the application
	$(CC) -g $(WARN) -O3 -o $(OUT) $(SOURCES) $(INCLUDES) -DNUMBER_OF_ACTIVATION_TABLE_VALUES=96 -DMIN_X_TABLE_VALUE=-6 -DMAX_X_TABLE_VALUE=6 -DACTIVATION_DELTA_SHIFT=3 -lstdc++fs

# Remove all object files
clean:
	rm -f ./output/*
