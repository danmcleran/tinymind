# Elman speaker recognition on UCI Japanese Vowels: offline float training,
# fixed-point (Q16.16 / Q8.8 / Q4.4) inference comparison.
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/elman_vowels
SOURCES=elman_vowels.cpp ../../cpp/lookupTables.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_HOSTED_RAND=1 \
	-DTINYMIND_USE_SIGMOID_16_16=1 -DTINYMIND_USE_TANH_16_16=1 \
	-DTINYMIND_USE_SIGMOID_8_8=1 -DTINYMIND_USE_TANH_8_8=1 \
	-DTINYMIND_USE_SIGMOID_4_4=1 -DTINYMIND_USE_TANH_4_4=1

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)
	cp ae.train ae.test ./output/

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)
	cp ae.train ae.test ./output/

run :
	cd ./output && ./elman_vowels && cd ../

plot :
	python3 plot.py

clean :
	rm -f ./output/*
