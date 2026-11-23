# Fixed-point log-mel spectrogram and MFCC front end (cpp/qmel.hpp) on a
# synthetic vibration recording, compared with the float definition.
# FLOAT + STD are on for the host-side table builders and the double
# reference; the front end itself is pure integer.
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/logmel_vibration
SOURCES=logmel_vibration.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_QUANTIZATION=1 -DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

run :
	cd ./output && ./logmel_vibration && cd ../

plot :
	python3 plot.py

clean :
	rm -f ./output/*
