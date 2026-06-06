# Closed-form Continuous-time (CfC) cell demo. The solver-free cell (cfc.hpp) is
# scalar-templated, so reverse-mode autodiff (revdual.hpp) trains it with no
# bespoke backprop. Float build: TINYMIND_ENABLE_FLOAT + STD (std::exp/tanh,
# float divide); the RevVar Constant<> lift is enabled by the example's
# -DTINYMIND_CFC_REVERSE_TRAINING and revdual's own FLOAT&&STD gate.

MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/cfc_sequence
SOURCES=cfc_sequence.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

run :
	cd ./output && ./cfc_sequence && cd ../

clean :
	rm -f ./output/*

plot :
	python3 plot.py
