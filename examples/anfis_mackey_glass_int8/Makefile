# int8 Mackey-Glass ANFIS. Runs the same frozen model through the float,
# Q16.16, and int8 tiers, and reports the memory each costs.
#
# The headline result is negative and deliberate: int8 is LARGER than Q16.16
# for this model, because cpp/qanfis.hpp evaluates membership functions
# through 256-entry lookup tables whose cost does not shrink with the
# parameter width they replace.
#
# FLOAT + STD are on because the example calibrates on the host (those helpers
# live behind that gate) and runs a double reference for comparison; the
# deployable int8 forward path itself needs neither.

MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/anfis_mackey_glass_int8
SOURCES=anfis_mackey_glass_int8.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_QUANTIZATION=1 -DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

run :
	cd ./output && ./anfis_mackey_glass_int8 && cd ../

golden :
	cd ./output && ./anfis_mackey_glass_int8 --golden > anfis_mackey_glass_int8.golden && cd ../

regenerate-model :
	python3 train.py

clean :
	rm -f ./output/*

plot :
	python3 plot.py
