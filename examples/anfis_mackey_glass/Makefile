# Mackey-Glass ANFIS demo (Jang's original 1993 benchmark for the
# architecture): predict x(t+6) from x(t-18), x(t-12), x(t-6), x(t) with a
# 16-rule Takagi-Sugeno system.
#
# cpp/anfis.hpp is inference only. The premise and consequent parameters are
# fitted on the host by apps/anfis_train (hybrid least squares + gradient
# descent) and committed as anfis_model.hpp, so the build needs no Python --
# `make regenerate-model` is a manual developer step, never part of CI.
#
# FLOAT + STD are on because the example runs a double reference alongside the
# Q16.16 deployment path and uses <cmath> for its error metrics. The shipped
# model itself uses generalized-bell membership functions, which are
# arithmetic-only and hold at FLOAT=0 / STD=0.

MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/anfis_mackey_glass
SOURCES=anfis_mackey_glass.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

run :
	cd ./output && ./anfis_mackey_glass && cd ../

regenerate-model :
	python3 train.py

# Deterministic Q16.16 byte stream for the unit_test/integration golden gate.
golden :
	cd ./output && ./anfis_mackey_glass --golden > anfis_mackey_glass.golden && cd ../

clean :
	rm -f ./output/*

plot :
	python3 plot.py
