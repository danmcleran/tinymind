# Int8 encoder-decoder (seq2seq) transformer. Exercises the decoder layer
# family (qcausalattention1d.hpp, qcrossattention.hpp, qkvcache.hpp) composed
# with the existing encoder primitives. The headline check is that the causal
# self-attention's autoregressive step() decode reproduces the full-sequence
# forward() byte-for-byte via the O(1) KV state.

MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/seq2seq_softmax_int8
SOURCES=seq2seq_softmax_int8.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_QUANTIZATION=1

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

run :
	cd ./output && ./seq2seq_softmax_int8 && cd ../

coverage :
	$(MKDIR)
	$(CC) --coverage -O0 -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

golden :
	cd ./output && ./seq2seq_softmax_int8 --golden > seq2seq_softmax_int8.golden && cd ../

clean :
	rm -f ./output/*

plot :
	python3 plot.py
