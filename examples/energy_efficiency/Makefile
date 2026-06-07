# Energy Efficiency regression (Q16.16 MLP) example Makefile.
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/energy_efficiency
SOURCES=energy_efficiency.cpp ../../cpp/lookupTables.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_HOSTED_IO=1 -DTINYMIND_ENABLE_HOSTED_RAND=1 -DTINYMIND_USE_SIGMOID_16_16=1

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)
	cp ENB2012_data.csv ./output/

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)
	cp ENB2012_data.csv ./output/

run :
	cd ./output && ./energy_efficiency && cd ../

plot :
	python3 plot.py

clean :
	rm -f ./output/*
