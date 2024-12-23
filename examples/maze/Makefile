# Simple Makefile for the table-based q-learner example
default :
#	Make an output dir to hold the executable
	mkdir -p ./output
#	Build the example with default build flags
	g++ -g -O3 -std=c++17 -Wall -o ./output/maze maze.cpp mazelearner.cpp -I../../cpp -I../../cpp/include

debug :
#	Make an output dir to hold the executable
	mkdir -p ./output
#	Build the example with default build flags
	g++ -g -std=c++17 -Wall -o ./output/maze maze.cpp mazelearner.cpp -I../../cpp -I../../cpp/include

# Remove all object files
clean:
	rm -f ./output/*
