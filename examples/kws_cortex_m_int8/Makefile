# int8 quantized variant of the KWS-style host runner. Uses the parallel
# Q* layer family (qconv2d, qdepthwiseconv2d, qpointwiseconv2d, qpool2d)
# plus the host-only calibration helpers in cpp/include/qcalibration.hpp.

MKDIR=mkdir -p ./output
CC=g++
WARN=-Wall -Wextra -Werror -Wpedantic
OUT=./output/kws_cortex_m_int8
SOURCES=kws_cortex_m_int8.cpp
INCLUDES=-I../../cpp -I../../cpp/include -I../../include/
DEFINES=-DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1 -DTINYMIND_ENABLE_QUANTIZATION=1

default :
	$(MKDIR)
	$(CC) -g $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

release :
	$(MKDIR)
	$(CC) -g -O3 $(WARN) -o $(OUT) $(SOURCES) $(INCLUDES) $(DEFINES)

run :
	cd ./output && ./kws_cortex_m_int8 && cd ../

clean :
	rm -f ./output/*
