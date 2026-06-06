# Phase 17 exemplar: hybrid int8 + Q-format inference-only MLP.
#   int8 QDense  ->  pure-integer affine->QValue bridge  ->  Q-format dense
#                ->  pure-integer QValue->affine bridge  ->  int8 QDense classifier.
# Targets the deployable freestanding shape; the Makefile builds it both
# hosted (for the parity report) and freestanding-clean (for regression).

MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/mixed_precision_mlp_int8_qformat
SOURCES=mixed_precision_mlp_int8_qformat.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_QUANTIZATION=1

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

run :
	cd ./output && ./mixed_precision_mlp_int8_qformat && cd ../

coverage :
	$(MKDIR)
	$(CC) --coverage -O0 -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

bench :
	cd ./output && ./mixed_precision_mlp_int8_qformat --bench > mixed_precision_mlp_int8_qformat.csv && cd ../

golden :
	cd ./output && ./mixed_precision_mlp_int8_qformat --golden > mixed_precision_mlp_int8_qformat.golden && cd ../

clean :
	rm -f ./output/*

plot :
	python3 plot.py
