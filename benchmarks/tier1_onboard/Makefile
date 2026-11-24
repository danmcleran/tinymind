# The Tier 1 work (#295 to #300) on the board: cycles per call for the
# generated digits model (int8, int4 dense layer, 16x8), a 256-sample log-mel
# frame and its MFCCs, k-means and Gaussian anomaly scores, and one k-means
# learning step. Shares the board support and the ST-LINK selection of
# ../framework_comparison_f411. See README.md.
#
# One workload group per image (IMG=int8 | int4 | 16x8 | features), so each
# fits the L031K6's 32 KB of flash:
#
#   make IMG=int8                build build/<BOARD>/<LEVEL>/<IMG>.elf
#   make IMG=int8 run            flash + capture -> results/raw_<IMG>_<BOARD>_<LEVEL>.txt
#   make BOARD=l031 LEVEL=O2 IMG=features run
#   make all-run                 every image on the selected board and level
#
# `run` fails unless every workload reproduces the host's output hashes.

BOARD ?= f411
LEVEL ?= O3
IMG   ?= int8
TM    ?= ../..
FC    := ../framework_comparison_f411
DIGITS := $(TM)/examples/onnx_digits_int8

ifeq ($(BOARD),f411)
  CPU        = -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16
  BOARD_SRC  = $(FC)/board/board.cpp
  LDSCRIPT   = $(FC)/board/stm32f411re.ld
  OCD_TARGET = target/stm32f4x.cfg
else ifeq ($(BOARD),l031)
  CPU        = -mcpu=cortex-m0plus -mthumb -mfloat-abi=soft
  BOARD_SRC  = $(FC)/board/board_l031k6.cpp
  LDSCRIPT   = $(FC)/board/stm32l031k6.ld
  OCD_TARGET = target/stm32l0.cfg
else
  $(error BOARD must be f411 or l031)
endif

ifeq ($(IMG),int8)
  WORK = WORK_INT8
else ifeq ($(IMG),int4)
  WORK = WORK_INT4
else ifeq ($(IMG),16x8)
  WORK = WORK_16X8
else ifeq ($(IMG),features)
  WORK = WORK_FEATURES
else
  $(error IMG must be int8, int4, 16x8 or features)
endif

# DSP=1 (F411 only) turns on the Arm DSP backend (SMLAD dot products),
# which the library leaves off by default; its own build directory and
# results file keep the two builds apart.
DSP ?= 0
ifeq ($(DSP),1)
  ifneq ($(BOARD),f411)
    $(error DSP=1 needs the Cortex-M4 (BOARD=f411); the Cortex-M0+ has no DSP extension)
  endif
  DSP_DEFS = -DTINYMIND_ENABLE_SIMD_ARM_DSP=1
  SUFFIX   = _dsp
else
  DSP_DEFS =
  SUFFIX   =
endif

# LAYERS=1 also times each layer of the digits models on its own. The extra
# copies of the layers do not fit the L031K6's flash, so it defaults on for
# the F411 only.
ifeq ($(BOARD),f411)
  LAYERS ?= 1
else
  LAYERS ?= 0
endif
ifeq ($(LAYERS),1)
  DSP_DEFS += -DTIME_LAYERS
  SUFFIX   := $(SUFFIX)_layers
endif

RAW := results/raw_$(IMG)_$(BOARD)_$(LEVEL)$(SUFFIX).txt
B   := build/$(BOARD)/$(LEVEL)$(SUFFIX)

# Probe serials (STLINK_f411 / STLINK_l031) live in the untracked local.mk
# next door; STLINK=<serial> on the command line overrides it.
-include $(FC)/local.mk
STLINK ?= $(STLINK_$(BOARD))
PORT   ?= $(firstword $(wildcard /dev/serial/by-id/usb-STMicroelectronics_*_STLink_$(STLINK)-if02))

CXX      = arm-none-eabi-g++
HOST_CXX ?= g++
SIZE     = arm-none-eabi-size
TM_INC   = -I$(TM)/cpp -I$(TM)/cpp/include -I$(TM)/include -I$(DIGITS) -DTINYMIND_ENABLE_QUANTIZATION=1
CXXFLAGS = $(CPU) -$(LEVEL) -g -ffunction-sections -fdata-sections -Wall -Wextra -Werror -std=c++14 \
           -fno-exceptions -fno-rtti -fno-threadsafe-statics -fno-use-cxa-atexit $(EXTRA)
LD       = -L$(FC)/board -T $(LDSCRIPT) -nostartfiles --specs=nano.specs --specs=nosys.specs -Wl,--gc-sections
TM_HDRS  = $(shell find $(TM)/cpp $(TM)/include -name '*.hpp' -o -name '*.h') $(wildcard $(DIGITS)/*.hpp)

all: $(B)/$(IMG).elf

build/gen_tables: gen_tables.cpp tier1_common.hpp $(TM_HDRS)
	@mkdir -p $(@D)
	$(HOST_CXX) -std=c++14 -O2 -Wall -Wextra -Werror $(TM_INC) -DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1 \
	    -I. gen_tables.cpp -o $@

build/tables.h: build/gen_tables $(TM)/examples/optical_digits/optdigits.tes
	./build/gen_tables $(TM)/examples/optical_digits/optdigits.tes $@

$(B)/$(IMG).elf: tier1_bench.cpp tier1_common.hpp build/tables.h $(BOARD_SRC) $(TM_HDRS) Makefile
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(DSP_DEFS) -D$(WORK) -I. -Ibuild -I$(FC)/board $(TM_INC) -o $@ $< $(BOARD_SRC) $(LD)
	@$(SIZE) $@ | tail -1

OPENOCD = openocd -f interface/stlink.cfg -c "adapter serial $(STLINK)" -f $(OCD_TARGET)
CAPTURE_MAX ?= 60

run: $(B)/$(IMG).elf
	@test -n "$(STLINK)" || { echo "no ST-LINK selected: set STLINK_$(BOARD) in $(FC)/local.mk or pass STLINK=<serial>"; exit 1; }
	@test -n "$(PORT)" || { echo "no VCP found for ST-LINK $(STLINK)"; exit 1; }
	@mkdir -p results
	@$(OPENOCD) -c "program $< verify exit" >/dev/null 2>&1 || { echo "flashing via ST-LINK $(STLINK) failed"; exit 1; }
	-@fuser -k $(PORT) >/dev/null 2>&1; stty -F $(PORT) 115200 raw -echo
	@( timeout $(CAPTURE_MAX) cat $(PORT) > $(B)/$(IMG).raw & ) ; sleep 0.5 ; \
	  $(OPENOCD) -c "init; reset run; shutdown" >/dev/null 2>&1 ; \
	  n=0; until tr -d '\r' < $(B)/$(IMG).raw | grep -q '^# end bench' || [ $$n -ge $$((2 * $(CAPTURE_MAX))) ]; do sleep 0.5; n=$$((n + 1)); done; \
	  fuser -k $(PORT) >/dev/null 2>&1; true
	@{ echo "img=$(IMG) board=$(BOARD) level=$(LEVEL) dsp=$(DSP) compiler=$$($(CXX) --version | head -1)"; \
	   echo "size $$($(SIZE) $< | tail -1)"; \
	   tr -d '\r' < $(B)/$(IMG).raw | sed -n '/^# begin bench/,/^# end bench/p'; } > $(B)/$(IMG).txt
	@grep -q '^# end bench' $(B)/$(IMG).txt || { echo "capture failed"; exit 1; }
	@cat $(B)/$(IMG).txt
	@! grep -Eq 'match=0' $(B)/$(IMG).txt || { echo "output hash mismatch: board differs from host"; exit 1; }
	@cp $(B)/$(IMG).txt $(RAW)

all-run:
	@for img in int8 int4 16x8 features; do $(MAKE) --no-print-directory IMG=$$img DSP=$(DSP) run || exit 1; done

clean:
	rm -rf build

.PHONY: all run all-run clean
