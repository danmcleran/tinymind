# Tiny autoregressive int8 text generation (decoder-only nano-LM). Drives the
# decoder family in its real loop: QCausalAttention1D::step() over a fixed
# E x E QLinearKVState, an output head, and argmax greedy decode. Shows that
# generation runs in O(1) attention memory and that the int8 greedy decode
# reproduces the float reference's tokens.

MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/tiny_generate_int8
SOURCES=tiny_generate_int8.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_QUANTIZATION=1

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

run :
	cd ./output && ./tiny_generate_int8 && cd ../

coverage :
	$(MKDIR)
	$(CC) --coverage -O0 -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

golden :
	cd ./output && ./tiny_generate_int8 --golden > tiny_generate_int8.golden && cd ../

clean :
	rm -f ./output/*

plot :
	python3 plot.py
