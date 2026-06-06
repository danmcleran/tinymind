# int8 Closed-form Continuous-time (CfC) liquid cell deployment exemplar.
# Calibrates a small CfC cell on the host, then runs a pure-integer recurrent
# forward through qcfc.hpp (QCfCCell). Mirrors the build pattern of the other
# int8 examples (e.g. examples/resnet_block_int8).

MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/qcfc_liquid_int8
SOURCES=qcfc_liquid_int8.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_QUANTIZATION=1

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

run :
	cd ./output && ./qcfc_liquid_int8 && cd ../

bench :
	cd ./output && ./qcfc_liquid_int8 --bench > qcfc_liquid_int8.csv && cat qcfc_liquid_int8.csv && cd ../

golden :
	cd ./output && ./qcfc_liquid_int8 --golden > qcfc_liquid_int8.golden && cat qcfc_liquid_int8.golden && cd ../

coverage :
	$(MKDIR)
	$(CC) --coverage -O0 -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

clean :
	rm -f ./output/*

plot :
	python3 plot.py
