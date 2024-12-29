# Simple Makefile for the activation function LUTs
default :
#	Make an output dir to hold the executable
	mkdir -p ./output
#	Build the application
	g++ -g -Wall -o ./output/activationTableGenerator activationTableGenerator.cpp -I../../cpp/ -lstdc++fs

# Remove all object files
clean:
	rm -f ./output/*
