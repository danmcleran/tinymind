# Handwritten-digit classifier (Q16.16 MLP) example Makefile.
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/optical_digits
SOURCES=optical_digits.cpp ../../cpp/lookupTables.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_HOSTED_IO=1 -DTINYMIND_ENABLE_HOSTED_RAND=1 -DTINYMIND_USE_SIGMOID_16_16=1

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)
	cp optdigits.tra optdigits.tes ./output/

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)
	cp optdigits.tra optdigits.tes ./output/

run :
	cd ./output && ./optical_digits && cd ../

plot :
	python3 plot.py

clean :
	rm -f ./output/*
