# Simple Makefile for the table-based q-learner example
default :
#	Make an output dir to hold the executable
	mkdir -p ./output
#	Build the example with default build flags
	g++ -O3 -Wall -o ./output/maze maze.cpp mazelearner.cpp -I../../cpp

debug :
#	Make an output dir to hold the executable
	mkdir -p ./output
#	Build the example with default build flags
	g++ -g -Wall -o ./output/maze maze.cpp mazelearner.cpp -I../../cpp

nonewlib :
#	Make an output dir to hold the executable
	mkdir -p ./output
#	Build the example with default build flags and do not include the new library
	g++ -O3 -Wall -o ./output/maze maze.cpp mazelearner.cpp -I../../cpp -DTINYMIND_DISABLE_NEW_LIB=1

# Remove all object files
clean:
	rm -f ./output/*
