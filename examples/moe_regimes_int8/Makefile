# int8 Mixture-of-Experts regime-routing exemplar.
# Approximates a nonlinear 1D target with a top-1 MoE of three linear experts
# deployed through cpp/qmoe.hpp::QMixtureOfExperts. Standalone (float weights
# fit in the binary), so `make check` runs it without numpy. FLOAT + STD on for
# the calibration helpers; QUANT=1 turns on the q*.hpp family.
MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/moe_regimes_int8
SOURCES=moe_regimes_int8.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_QUANTIZATION=1

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

run :
	cd ./output && ./moe_regimes_int8 && cd ../

bench :
	cd ./output && ./moe_regimes_int8 --bench && cd ../

golden :
	cd ./output && ./moe_regimes_int8 --golden > moe_regimes_int8.golden && cat moe_regimes_int8.golden && cd ../

coverage :
	$(MKDIR)
	$(CC) --coverage -O0 -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

clean :
	rm -f ./output/*

plot :
	python3 plot.py
