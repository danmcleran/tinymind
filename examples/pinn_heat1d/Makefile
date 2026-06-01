# PINN building block: 1-D heat-equation residual via forward-mode autodiff,
# plus the primary deployment path (offline-trained field, fixed-point
# inference). The double/autodiff path uses std::tanh; the Q16.16 inference
# path uses the tanh lookup table, hence lookupTables.cpp + TINYMIND_USE_TANH_16_16.

MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/pinn_heat1d
SOURCES=pinn_heat1d.cpp ../../cpp/lookupTables.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1 -DTINYMIND_USE_TANH_16_16=1

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

run :
	cd ./output && ./pinn_heat1d && cd ../

clean :
	rm -f ./output/*
