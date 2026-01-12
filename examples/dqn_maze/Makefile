# Simple Makefile for the DQN maze example
# Tell the code to compile the Q16.16 tanh activation function LUT
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/dqn_maze
SOURCES=dqn_maze.cpp dqn_mazelearner.cpp ../../cpp/lookupTables.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_USE_TANH_16_16=1

default:
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)
	$(CC) -c dqn_mazelearner.cpp $(WARN) -o output/dqn_mazelearner.o $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)
	$(CC) -c dqn_mazelearner.cpp -O3 $(WARN) -o output/dqn_mazelearner.o $(INCLUDES) $(DEFINES)

run :
	cd ./output && ./dqn_maze && cd ../

plot :
	python3 ./dqn_mazeplot.py output/dqn_maze_training.txt &
	python3 ./dqn_mazeplot.py output/dqn_maze_test.txt &

clean:
	rm -f ./output/*
