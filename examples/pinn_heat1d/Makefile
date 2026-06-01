# PINN building block: 1-D heat-equation residual via forward-mode autodiff.
# Float/double path (the recommended PINN proving regime), so the dual
# activations resolve to std::tanh and no lookup-table object is needed.

MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/pinn_heat1d
SOURCES=pinn_heat1d.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1

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
