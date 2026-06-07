# Human Activity Recognition (HAR) recurrent LSTM example Makefile.
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/har_activity
SOURCES=har_activity.cpp ../../cpp/lookupTables.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_HOSTED_IO=1 -DTINYMIND_ENABLE_HOSTED_RAND=1 -DTINYMIND_USE_SIGMOID_16_16=1 -DTINYMIND_USE_TANH_16_16=1

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

run :
	cd ./output && ./har_activity && cd ../

plot :
	python3 plot.py

clean :
	rm -f ./output/*
