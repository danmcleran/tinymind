# Full int8 transformer encoder STACK: token embedding + sinusoidal
# positional encoding + NUM_BLOCKS stacked encoder blocks. Exercises the
# new qembedding.hpp / qpositional.hpp primitives composed with the
# existing Q* layer family. Mirrors examples/transformer_encoder_int8.

MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/transformer_encoder_stack_int8
SOURCES=transformer_encoder_stack_int8.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_QUANTIZATION=1

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

run :
	cd ./output && ./transformer_encoder_stack_int8 && cd ../

coverage :
	$(MKDIR)
	$(CC) --coverage -O0 -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

golden :
	cd ./output && ./transformer_encoder_stack_int8 --golden > transformer_encoder_stack_int8.golden && cd ../

clean :
	rm -f ./output/*

plot :
	python3 plot.py
