# Simple Makefile for the table-based q-learner example
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
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

run :
	cd ./output && ./maze && cd ../

plot :
#	Plot the training and test results
	python3 ./mazeplot.py output/maze_training.txt &
	python3 ./mazeplot.py output/maze_test.txt &

# Remove all object files
clean:
	rm -f ./output/*
