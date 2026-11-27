# TFLite -> tinymind exemplar: a residual int8 CNN trained in Keras, converted
# by TFLiteConverter and run from the model.hpp that apps/import_tflite
# generated; every output must equal TFLM's. The .tflite model and the
# generated headers are committed; `make regenerate` rebuilds them (needs a
# Python with numpy, tensorflow and tflite-micro in a venv, never the system
# Python).
#
# Freestanding library configuration: FLOAT=0 and STD=0. The generated model
# needs neither; only this driver's file I/O and timer are hosted.
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/tflite_digits_int8
SOURCES=tflite_digits_int8.cpp
INCLUDES=-I. -I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_QUANTIZATION=1 -DTINYMIND_ENABLE_FLOAT=0 -DTINYMIND_ENABLE_STD=0
PY ?= python3

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

run :
	cd ./output && ./tflite_digits_int8 && cd ../

bench :
	cd ./output && ./tflite_digits_int8 --bench > tflite_digits_int8_bench.csv && cd ../

golden :
	cd ./output && ./tflite_digits_int8 --golden > tflite_digits_int8.golden && cd ../

regenerate :
	$(PY) -I train.py

plot :
	python3 plot.py

clean :
	rm -f ./output/*
