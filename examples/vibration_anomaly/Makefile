# On-device vibration anomaly detection: log-mel features (cpp/qmel.hpp) and
# a k-means baseline learned frame by frame (cpp/qanomaly.hpp), against a
# host-fitted diagonal Gaussian. FLOAT + STD are on for the table builders,
# the Gaussian fit and the metrics; the detection path is pure integer.
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/vibration_anomaly
SOURCES=vibration_anomaly.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_QUANTIZATION=1 -DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

run :
	cd ./output && ./vibration_anomaly && cd ../

plot :
	python3 plot.py

clean :
	rm -f ./output/*
