# Simple Makefile for the table-based q-learner example
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/maze
SOURCES=maze.cpp mazelearner.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES)
	$(CC) -c mazelearner.cpp $(WARN) -o output/mazelearner.o $(INCLUDES)

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES)
	$(CC) -c mazelearner.cpp -O3 $(WARN) -o output/mazelearner.o $(INCLUDES)

run :
	cd ./output && ./maze && cd ../

plot :
	python3 ./mazeplot.py output/maze_training.txt &
	python3 ./mazeplot.py output/maze_test.txt &

clean:
	rm -f ./output/*
