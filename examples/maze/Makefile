# Simple Makefile for the table-based q-learner example
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror
OUT=./output/maze
SOURCES=maze.cpp mazelearner.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/

default :
#	Make an output dir to hold the executable
	$(MKDIR)
#	Build the example with default build flags
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES)

debug :
#	Make an output dir to hold the executable
	$(MKDIR)
#	Build the example with default build flags
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES)

# Remove all object files
clean:
	rm -f ./output/*
