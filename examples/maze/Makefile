# Simple Makefile for the table-based q-learner example
default :
#	Make an output dir to hold the executable
	mkdir -p ./output
#	Build the example with default build flags
	g++ -g -O3 -Wall -o ./output/maze maze.cpp mazelearner.cpp -I../../cpp -I../../cpp/include

debug :
#	Make an output dir to hold the executable
	mkdir -p ./output
#	Build the example with default build flags
	g++ -g -Wall -o ./output/maze maze.cpp mazelearner.cpp -I../../cpp -I../../cpp/include

# Remove all object files
clean:
	rm -f ./output/*
