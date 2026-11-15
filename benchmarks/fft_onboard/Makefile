# QFFT1D vs QRFFT1D on the board: cycles per forward transform on real
# input, N = 64 .. 1024 (512 on the L031). Shares the board support and the
# ST-LINK selection of ../framework_comparison_f411. See README.md.
#
#   make                         build build/<BOARD>/<LEVEL>/fft.elf
#   make run                     flash + capture -> results/raw_<BOARD>_<LEVEL>.txt
#   make BOARD=l031 LEVEL=O2 run Cortex-M0+ NUCLEO-L031K6, -O2
#   make TW=ram run              twiddle tables in RAM instead of flash
#                                (results/raw_<BOARD>_<LEVEL>_twram.txt)
#
# `run` fails unless every transform reproduces the host's output hashes.

BOARD ?= f411
LEVEL ?= O3
TW    ?= flash
TM    ?= ../..
FC    := ../framework_comparison_f411

ifeq ($(BOARD),f411)
  CPU        = -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16
  BOARD_SRC  = $(FC)/board/board.cpp
  LDSCRIPT   = $(FC)/board/stm32f411re.ld
  OCD_TARGET = target/stm32f4x.cfg
  MAXN       = 1024
else ifeq ($(BOARD),l031)
  CPU        = -mcpu=cortex-m0plus -mthumb -mfloat-abi=soft
  BOARD_SRC  = $(FC)/board/board_l031k6.cpp
  LDSCRIPT   = $(FC)/board/stm32l031k6.ld
  OCD_TARGET = target/stm32l0.cfg
  # Three 1024-entry int16 buffers would take 6 KB of the L031's 8 KB.
  MAXN       = 512
else
  $(error BOARD must be f411 or l031)
endif

ifeq ($(TW),ram)
  TW_DEFS = -DTW_IN_RAM
  SUFFIX  = _twram
else ifeq ($(TW),flash)
  TW_DEFS =
  SUFFIX  =
else
  $(error TW must be flash or ram)
endif

B   := build/$(BOARD)/$(LEVEL)$(SUFFIX)
RAW := results/raw_$(BOARD)_$(LEVEL)$(SUFFIX).txt

# Probe serials (STLINK_f411 / STLINK_l031) live in the untracked local.mk
# next door; STLINK=<serial> on the command line overrides it.
-include $(FC)/local.mk
STLINK ?= $(STLINK_$(BOARD))
PORT   ?= $(firstword $(wildcard /dev/serial/by-id/usb-STMicroelectronics_*_STLink_$(STLINK)-if02))

CXX      = arm-none-eabi-g++
HOST_CXX ?= g++
SIZE     = arm-none-eabi-size
TM_INC   = -I$(TM)/cpp -I$(TM)/cpp/include -I$(TM)/include -DTINYMIND_ENABLE_QUANTIZATION=1
CXXFLAGS = $(CPU) -$(LEVEL) -g -ffunction-sections -fdata-sections -Wall -Wextra -Werror -std=c++14 \
           -fno-exceptions -fno-rtti -fno-threadsafe-statics -fno-use-cxa-atexit
LD       = -L$(FC)/board -T $(LDSCRIPT) -nostartfiles --specs=nano.specs --specs=nosys.specs -Wl,--gc-sections
TM_HDRS  = $(shell find $(TM)/cpp $(TM)/include -name '*.hpp' -o -name '*.h')

all: $(B)/fft.elf

build/gen_tw: gen_tw.cpp fft_common.hpp $(TM_HDRS)
	@mkdir -p $(@D)
	$(HOST_CXX) -std=c++14 -O2 -Wall -Wextra -Werror $(TM_INC) -DTINYMIND_ENABLE_FLOAT=1 -DTINYMIND_ENABLE_STD=1 \
	    -I. gen_tw.cpp -o $@

build/tw.h: build/gen_tw
	./build/gen_tw $@

$(B)/fft.elf: fft_bench.cpp fft_common.hpp build/tw.h $(BOARD_SRC) $(TM_HDRS) Makefile
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -DMAXN=$(MAXN) $(TW_DEFS) -I. -Ibuild -I$(FC)/board $(TM_INC) \
	    -o $@ fft_bench.cpp $(BOARD_SRC) $(LD)
	@$(SIZE) $@ | tail -1

OPENOCD = openocd -f interface/stlink.cfg -c "adapter serial $(STLINK)" -f $(OCD_TARGET)
CAPTURE_MAX ?= 60

run: $(B)/fft.elf
	@test -n "$(STLINK)" || { echo "no ST-LINK selected: set STLINK_$(BOARD) in $(FC)/local.mk or pass STLINK=<serial>"; exit 1; }
	@test -n "$(PORT)" || { echo "no VCP found for ST-LINK $(STLINK)"; exit 1; }
	@mkdir -p results
	@$(OPENOCD) -c "program $< verify exit" >/dev/null 2>&1 || { echo "flashing via ST-LINK $(STLINK) failed"; exit 1; }
	-@fuser -k $(PORT) >/dev/null 2>&1; stty -F $(PORT) 115200 raw -echo
	@( timeout $(CAPTURE_MAX) cat $(PORT) > $(B)/fft.raw & ) ; sleep 0.5 ; \
	  $(OPENOCD) -c "init; reset run; shutdown" >/dev/null 2>&1 ; \
	  n=0; until tr -d '\r' < $(B)/fft.raw | grep -q '^# end bench' || [ $$n -ge $$((2 * $(CAPTURE_MAX))) ]; do sleep 0.5; n=$$((n + 1)); done; \
	  fuser -k $(PORT) >/dev/null 2>&1; true
	@{ echo "board=$(BOARD) level=$(LEVEL) compiler=$$($(CXX) --version | head -1)"; \
	   tr -d '\r' < $(B)/fft.raw | sed -n '/^# begin bench/,/^# end bench/p'; } > $(B)/fft.txt
	@grep -q '^# end bench' $(B)/fft.txt || { echo "capture failed"; exit 1; }
	@cat $(B)/fft.txt
	@! grep -Eq 'q(r)?fft_match=0' $(B)/fft.txt || { echo "output hash mismatch: board differs from host"; exit 1; }
	@cp $(B)/fft.txt $(RAW)

clean:
	rm -rf build

.PHONY: all run clean
