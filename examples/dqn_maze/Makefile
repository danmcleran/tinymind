# Simple Makefile for the DQN maze example
# Tell the code to compile the Q16.16 tanh activation function LUT
default:
#	Make an output dir to hold the executable
	mkdir -p ./output
#	Build the example with default build flags
	g++ -g -O3 -Wall -o ./output/dqn_maze dqn_maze.cpp dqn_mazelearner.cpp ../../cpp/lookupTables.cpp -I../../cpp -I../../cpp/include -I../../include/ -DTINYMIND_USE_TANH_16_16=1

debug:
#	Make an output dir to hold the executable
	mkdir -p ./output
#	Build the example with default build flags
	g++ -g -Wall -o ./output/dqn_maze dqn_maze.cpp dqn_mazelearner.cpp ../../cpp/lookupTables.cpp -I../../cpp -I../../cpp/include -I../../include/ -DTINYMIND_USE_TANH_16_16=1

# Remove all object files
clean:
	rm -f ./output/*
