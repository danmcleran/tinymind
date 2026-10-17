# Cross-framework NN benchmark on NUCLEO-F411RE (Cortex-M4F @ 100 MHz):
# tinymind vs TensorFlow Lite for Microcontrollers (CMSIS-NN and reference
# kernels) vs emlearn. See README.md for method and results.
#
# BOARD=l031 retargets every target below at a NUCLEO-L031K6 (Cortex-M0+ @
# 32 MHz, 32 KB flash, 8 KB SRAM): own build/l031/ and results/raw_l031_*
# files, and a 32-sample MLP test set so the images fit in flash.
#
#   make deps                 venv (TF, emlearn, sklearn) + TFLM clone + TFLM libs
#   make host-check           tinymind int8 outputs reproduce the expected hashes
#   make host-bench           time tinymind on the build host, scalar and each x86
#                             SIMD backend the CPU supports -> results/host_x86.txt
#   make LEVEL=O2|O3          build every image into build/<LEVEL>/
#   make LEVEL=O2|O3 run      flash + capture each image -> results/raw_<LEVEL>.txt
#   make LEVEL=.. IMG=x run1  flash + capture one image
#   make probe                bisect the minimum TFLM tensor arena on the board
#   make BOARD=l031 fit       flash/RAM the images that do not fit the L031 would need
#   make stlink               list attached ST-LINK probes and which one is used
#   make models               regenerate gen/out/ (needs the venv)
#
# LEVEL applies to every framework at once: tinymind and emlearn translation
# units get -O2/-O3; TFLM links libs built with its default release flags
# (O2: kernels -O2, core -Os) or with every level forced to -O3 (O3).
#
# Lives outside examples/: the top-level build and check-clang-examples build
# every Makefile under examples/ with the host compiler, and this one targets Arm.

LEVEL   ?= O2
BOARD   ?= f411
TM      ?= ../..
TFLM    ?= tflite-micro
TFLM_REV ?= 18811372beeed5d8a05f9b356f079a369c099540
PY      ?= .venv/bin/python
ifeq ($(LEVEL),O3)
  OPT := -O3
  TFLM_GEN := $(TFLM)/gen_o3
else
  OPT := -O2
  TFLM_GEN := $(TFLM)/gen
endif
TFLM_ARCH_f411 := cortex-m4+fp
TFLM_ARCH_l031 := cortex-m0plus
TFLM_ARCH  := $(TFLM_ARCH_$(BOARD))
TFLM_CMSIS := $(TFLM_GEN)/cortex_m_generic_$(TFLM_ARCH)_release_cmsis_nn_gcc/lib/libtensorflow-microlite.a
TFLM_REF   := $(TFLM_GEN)/cortex_m_generic_$(TFLM_ARCH)_release_gcc/lib/libtensorflow-microlite.a
TFLM_LOGS  := $(TFLM_GEN)/cortex_m_generic_$(TFLM_ARCH)_release_with_logs_cmsis_nn_gcc/lib/libtensorflow-microlite.a
EML_INC ?= $(shell $(PY) -I -c "import emlearn; print(emlearn.includedir)" 2>/dev/null)

ifeq ($(BOARD),f411)
  CPU      = -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16
  BOARD_SRC = board/board.cpp
  LDSCRIPT = board/stm32f411re.ld
  OCD_TARGET = target/stm32f4x.cfg
  B       := build/$(LEVEL)
  RAW      = results/raw_$(LEVEL).txt
  DATA_INC =
  BOARD_DEFS =
else ifeq ($(BOARD),l031)
  CPU      = -mcpu=cortex-m0plus -mthumb -mfloat-abi=soft
  BOARD_SRC = board/board_l031k6.cpp
  LDSCRIPT = board/stm32l031k6.ld
  OCD_TARGET = target/stm32l0.cfg
  B       := build/l031/$(LEVEL)
  RAW      = results/raw_l031_$(LEVEL).txt
  # 360 MLP samples are 23 KB of int8 inputs (92 KB as float) alone; keep the
  # first 32, so the float emlearn image fits too.
  MLP_SUBSET = 32
  FLASH_BYTES = 32768
  RAM_BYTES = 8192
  DATA_INC = -I$(B)/data
  BOARD_DEFS = -DBENCH_SMALL_RAM
else
  $(error BOARD must be f411 or l031)
endif

# Which ST-LINK to flash and read. With one probe attached it is found on its
# own; with several, STLINK=<serial> picks one (`make stlink` lists them), on
# the command line, in the environment, or in an untracked local.mk (which can
# also hold one STLINK_<board> per board, used when STLINK is unset). Both
# openocd and the VCP port follow the serial, so the two cannot disagree and
# replugging (which renumbers /dev/ttyACM*) does not matter.
-include local.mk
STLINK ?= $(STLINK_$(BOARD))
STLINK_VCPS := $(sort $(wildcard /dev/serial/by-id/usb-STMicroelectronics_*_STLink_*-if02))
STLINK_ALL  := $(foreach v,$(STLINK_VCPS),$(lastword $(subst _, ,$(v:%-if02=%))))
ifeq ($(STLINK),)
  ifeq ($(words $(STLINK_ALL)),1)
    STLINK := $(STLINK_ALL)
  endif
endif
PORT    ?= $(firstword $(filter %_$(STLINK)-if02,$(STLINK_VCPS)))

CXX  = arm-none-eabi-g++
CC   = arm-none-eabi-gcc
SIZE = arm-none-eabi-size
COMMON = $(CPU) $(OPT) -g -ffunction-sections -fdata-sections -Wall -Wextra -Werror
CXXFLAGS = $(COMMON) -std=c++17 -fno-exceptions -fno-rtti -fno-threadsafe-statics -fno-use-cxa-atexit
CFLAGS   = $(COMMON) -std=c99
INC  = -Iboard -Ifw $(DATA_INC) -Igen/out $(BOARD_DEFS)
LD   = -Lboard -T $(LDSCRIPT) -nostartfiles --specs=nano.specs --specs=nosys.specs -Wl,--gc-sections
HARNESS = $(BOARD_SRC) fw/harness.cpp

TM_INC   = -I$(TM)/cpp -I$(TM)/cpp/include -I$(TM)/include -DTINYMIND_ENABLE_QUANTIZATION=1
DSP      = -DTINYMIND_ENABLE_SIMD_ARM_DSP=1
TFLM_INC = -isystem $(TFLM) -isystem $(TFLM)/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
           -isystem $(TFLM)/tensorflow/lite/micro/tools/make/downloads/gemmlowp \
           -DTF_LITE_STRIP_ERROR_STRINGS -DNDEBUG -funsigned-char

ifeq ($(BOARD),f411)
IMAGES = baseline \
  tm_mlp tm_mlp_dsp tm_kws tm_kws_dsp \
  tflm_cmsis_mlp_int8 tflm_ref_mlp_int8 tflm_cmsis_mlp_f32 \
  tflm_cmsis_kws_int8 tflm_ref_kws_int8 tflm_cmsis_kws_f32 \
  eml_mlp
PROF_IMAGES = tm_mlp_prof tm_mlp_dsp_prof tm_kws_prof tm_kws_dsp_prof \
  tflm_cmsis_mlp_int8_prof tflm_cmsis_kws_int8_prof
else
# ARMv6-M has no SMLAD, so no +dsp images (the ARM_DSP gate #errors there).
# tm_rf / eml_rf_*: integer random forest, the one model emlearn also runs
# integer-only (it has no int8 MLP); see gen/make_trees.py. tm_rf walks the
# QTreeNode pool; tm_rf_inline{,_u8} expand it at compile time (QStaticDecisionTree).
IMAGES = baseline tm_mlp tm_kws eml_mlp tm_rf tm_rf_inline tm_rf_inline_u8 eml_rf_inline eml_rf_loadable
PROF_IMAGES = tm_mlp_prof tm_kws_prof
# Images that do not fit the L031. `make BOARD=l031 fit` links them against an
# unbounded memory map (board/unbounded.ld) to report what they would need.
FIT_IMAGES = tflm_cmsis_mlp_int8 tflm_ref_mlp_int8 tflm_cmsis_mlp_f32 \
  tflm_cmsis_kws_int8 tflm_ref_kws_int8 tflm_cmsis_kws_f32
endif

# Smallest tensor arena AllocateTensors() accepts, bisected on the board with
# `make probe` (arena_used_bytes() under-reports it by 100-300 B).
ARENA_tflm_cmsis_mlp_int8 = 1680
ARENA_tflm_ref_mlp_int8   = 1584
ARENA_tflm_cmsis_mlp_f32  = 1200
ARENA_tflm_cmsis_kws_int8 = 4800
ARENA_tflm_ref_kws_int8   = 4736
ARENA_tflm_cmsis_kws_f32  = 14112

all: $(IMAGES:%=$(B)/%.elf) $(PROF_IMAGES:%=$(B)/%.elf)

$(B):
	mkdir -p $@

ifdef MLP_SUBSET
$(B)/data/%_data.h: gen/out/%_data.h gen/subset.py
	@mkdir -p $(@D)
	python3 -I gen/subset.py $< $@ $(MLP_SUBSET)
DATA_DEPS = $(B)/data/mlp_data.h $(B)/data/rf_data.h
endif

# emlearn's generated MLP ends in softmax; patch it to identity so it returns
# logits like the other backends (argmax is unchanged).
$(B)/mlp_emlearn_logits.h: gen/out/mlp_emlearn.h | $(B) $(DATA_DEPS)
	sed 's/EmlNetActivationSoftmax/EmlNetActivationIdentity/' $< > $@

$(B)/baseline.elf: fw/be_baseline.cpp $(HARNESS) | $(B) $(DATA_DEPS)
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $^ $(LD)

# $(1)=image $(2)=extra defines
define tm_rule
$(B)/$(1).elf: fw/be_tinymind.cpp $$(HARNESS) | $$(B) $$(DATA_DEPS)
	$$(CXX) $$(CXXFLAGS) $$(INC) $$(TM_INC) $(2) -o $$@ $$^ $$(LD)
endef
$(eval $(call tm_rule,tm_mlp,-DMODEL_MLP))
$(eval $(call tm_rule,tm_mlp_dsp,-DMODEL_MLP $(DSP)))
$(eval $(call tm_rule,tm_kws,-DMODEL_KWS))
$(eval $(call tm_rule,tm_kws_dsp,-DMODEL_KWS $(DSP)))
$(eval $(call tm_rule,tm_mlp_prof,-DMODEL_MLP -DPROFILE))
$(eval $(call tm_rule,tm_mlp_dsp_prof,-DMODEL_MLP $(DSP) -DPROFILE))
$(eval $(call tm_rule,tm_kws_prof,-DMODEL_KWS -DPROFILE))
$(eval $(call tm_rule,tm_kws_dsp_prof,-DMODEL_KWS $(DSP) -DPROFILE))

# $(1)=image $(2)=lib $(3)=defines $(4)=kernels tag
define tflm_rule
$(B)/$(1).elf: fw/be_tflm.cpp $$(HARNESS) $(2) | $$(B) $$(DATA_DEPS)
	$$(CXX) $$(CXXFLAGS) $$(INC) $$(TFLM_INC) $(3) -DKERNELS_TAG='"$(4)"' -DARENA_BYTES=$$(ARENA_$(1)) -o $$@ fw/be_tflm.cpp $$(HARNESS) $(2) -lm $$(LD)
$(B)/$(1)_prof.elf: fw/be_tflm.cpp $$(HARNESS) $$(TFLM_LOGS) | $$(B) $$(DATA_DEPS)
	$$(CXX) $$(CXXFLAGS) $$(INC) $$(TFLM_INC) -UTF_LITE_STRIP_ERROR_STRINGS -DPROFILE $(3) -DKERNELS_TAG='"$(4)"' -DARENA_BYTES=$$(ARENA_$(1)) -o $$@ fw/be_tflm.cpp $$(HARNESS) $$(TFLM_LOGS) -lm $$(LD)
$(B)/$(1)_probe.elf: fw/be_tflm.cpp $$(HARNESS) $(2) | $$(B) $$(DATA_DEPS)
	$$(CXX) $$(CXXFLAGS) $$(INC) $$(TFLM_INC) -DARENA_PROBE $(3) -DKERNELS_TAG='"$(4)"' -DARENA_BYTES=32768 -o $$@ fw/be_tflm.cpp $$(HARNESS) $(2) -lm $$(LD)
endef
$(eval $(call tflm_rule,tflm_cmsis_mlp_int8,$(TFLM_CMSIS),-DMODEL_MLP,+cmsis))
$(eval $(call tflm_rule,tflm_ref_mlp_int8,$(TFLM_REF),-DMODEL_MLP,+ref))
$(eval $(call tflm_rule,tflm_cmsis_mlp_f32,$(TFLM_CMSIS),-DMODEL_MLP -DDTYPE_F32,+cmsis))
$(eval $(call tflm_rule,tflm_cmsis_kws_int8,$(TFLM_CMSIS),-DMODEL_KWS,+cmsis))
$(eval $(call tflm_rule,tflm_ref_kws_int8,$(TFLM_REF),-DMODEL_KWS,+ref))
$(eval $(call tflm_rule,tflm_cmsis_kws_f32,$(TFLM_CMSIS),-DMODEL_KWS -DDTYPE_F32,+cmsis))

# $(1)=image $(2)=extra defines
define tm_rf_rule
$(B)/$(1).elf: fw/be_tinymind_rf.cpp $$(HARNESS) | $$(B) $$(DATA_DEPS)
	$$(CXX) $$(CXXFLAGS) $$(INC) $$(TM_INC) $(2) -o $$@ $$^ $$(LD)
endef
$(eval $(call tm_rf_rule,tm_rf,))
$(eval $(call tm_rf_rule,tm_rf_inline,-DTM_RF_INLINE))
$(eval $(call tm_rf_rule,tm_rf_inline_u8,-DTM_RF_INLINE -DTM_RF_U8))

# $(1)=image $(2)=extra defines
define eml_rf_rule
$(B)/$(1).o: fw/be_emlearn_rf.c | $$(B) $$(DATA_DEPS)
	@test -n "$$(EML_INC)" || { echo "emlearn not found: run 'make deps' or pass EML_INC=<emlearn include dir>"; exit 1; }
	$$(CC) $$(CFLAGS) $$(INC) -isystem $$(EML_INC) $(2) -c -o $$@ $$<
$(B)/$(1).elf: $(B)/$(1).o $$(HARNESS) | $$(B)
	$$(CXX) $$(CXXFLAGS) $$(INC) -o $$@ $$^ $$(LD)
endef
$(eval $(call eml_rf_rule,eml_rf_inline,))
$(eval $(call eml_rf_rule,eml_rf_loadable,-DEML_RF_LOADABLE))

$(B)/eml_mlp.o: fw/be_emlearn.c $(B)/mlp_emlearn_logits.h | $(B) $(DATA_DEPS)
	@test -n "$(EML_INC)" || { echo "emlearn not found: run 'make deps' or pass EML_INC=<emlearn include dir>"; exit 1; }
	$(CC) $(CFLAGS) $(INC) -I$(B) -isystem $(EML_INC) -c -o $@ $<
$(B)/eml_mlp.elf: $(B)/eml_mlp.o $(HARNESS) | $(B) $(DATA_DEPS)
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $^ -lm $(LD)

OPENOCD = openocd -f interface/stlink.cfg -c "adapter serial $(STLINK)" -f $(OCD_TARGET)

stlink:
	@for v in $(STLINK_VCPS); do s=$${v%-if02}; echo "$${s##*_} $$(readlink -f $$v)"; done
	@echo "using: $(if $(STLINK),$(STLINK) on $(PORT),none)"

# Refuse to guess: with no probe, or several and no STLINK, any flash could hit the wrong board.
stlink-check:
	@test -n "$(STLINK)" || { echo "no ST-LINK selected: $(words $(STLINK_ALL)) attached ($(STLINK_ALL)); pass STLINK=<serial>"; exit 1; }
	@test -n "$(PORT)" || { echo "no VCP found for ST-LINK $(STLINK); check 'make stlink'"; exit 1; }

run1: stlink-check
	@mkdir -p results
	@$(OPENOCD) -c "program $(B)/$(IMG).elf verify exit" >/dev/null 2>&1 || \
	  { echo "flashing $(IMG) via ST-LINK $(STLINK) failed: is that probe on the $(BOARD)?"; exit 1; }
	-fuser -k $(PORT) >/dev/null 2>&1; stty -F $(PORT) 115200 raw -echo
	( timeout 8 cat $(PORT) > $(B)/$(IMG).raw & ) ; sleep 0.5 ; \
	  $(OPENOCD) -c "init; reset run; shutdown" >/dev/null 2>&1 ; sleep 6
	tr -d '\r' < $(B)/$(IMG).raw | sed -n '/^# begin bench/,/^# end bench/p' > $(B)/$(IMG).txt
	@grep -q '^# end bench' $(B)/$(IMG).txt || { echo "capture failed for $(IMG)"; exit 1; }
	@cat $(B)/$(IMG).txt

run: all stlink-check
	@rm -f $(RAW)
	@for i in $(IMAGES) $(PROF_IMAGES); do $(MAKE) -s BOARD=$(BOARD) LEVEL=$(LEVEL) STLINK=$(STLINK) IMG=$$i run1 > /dev/null && \
	  { echo "level=$(LEVEL)"; echo "image=$$i"; cat $(B)/$$i.txt; } >> $(RAW) || exit 1; done
	@for i in $(IMAGES); do $(call size_line,$(B)/$$i.elf,$$i,); done >> $(RAW)
ifdef FIT_IMAGES
	@$(MAKE) -s BOARD=$(BOARD) LEVEL=$(LEVEL) fit >> $(RAW)
endif

# $(1)=elf $(2)=image name $(3)=extra awk printf suffix
size_line = $(SIZE) -A $(1) | awk -v n=$(2) '/^\.(isr_vector|text|rodata|ARM|init_array|data|bss)/{s[$$1]+=$$2} END{printf "size image=%s flash=%d ram=%d$(3)\n", n, s[".isr_vector"]+s[".text"]+s[".rodata"]+s[".ARM.exidx"]+s[".init_array"]+s[".data"], s[".data"]+s[".bss"]}'

# Same images, same flags, linked with no memory limit: flash and static RAM
# they would need. TFLM's tensor arena is part of .bss, so it is counted.
FB := $(B)/fit
fit: $(FIT_IMAGES:%=$(FB)/%.elf)
	@for i in $(FIT_IMAGES); do $(call size_line,$(FB)/$$i.elf,$$i,) | \
	  awk '{split($$3,f,"="); split($$4,r,"="); print $$0 " fits=" (f[2] <= $(FLASH_BYTES) && r[2] <= $(RAM_BYTES) ? "yes" : "no")}'; done
$(FB)/%.elf: | $(B) $(DATA_DEPS)
	@mkdir -p $(FB)
	$(MAKE) -s BOARD=$(BOARD) LEVEL=$(LEVEL) B=$(FB) LDSCRIPT=board/unbounded.ld DATA_INC="$(DATA_INC)" $@

PROBES = tflm_cmsis_mlp_int8 tflm_ref_mlp_int8 tflm_cmsis_mlp_f32 tflm_cmsis_kws_int8 tflm_ref_kws_int8 tflm_cmsis_kws_f32
probe: $(PROBES:%=$(B)/%_probe.elf) stlink-check
	@for i in $(PROBES); do $(MAKE) -s BOARD=$(BOARD) LEVEL=$(LEVEL) STLINK=$(STLINK) IMG=$${i}_probe run1 | grep -E "^(name|arena_used|setup_rc)"; done

host-check:
	@mkdir -p build
	g++ -std=c++17 -O2 -Wall -Wextra -Werror -DTINYMIND_ENABLE_QUANTIZATION=1 -Ifw -Igen/out \
	    -I$(TM)/cpp -I$(TM)/cpp/include -I$(TM)/include host/tm_check.cpp -o build/tm_check
	./build/tm_check

# Host timing of the same generated models, once per x86 SIMD backend and
# level. Gates are cumulative, as in a build for a CPU that has them all.
# Every backend is compiled; one the CPU lacks (per /proc/cpuinfo) is skipped
# rather than run, since it would die on an illegal instruction. HOST_PIN pins
# each run to one core when taskset is available.
HOST_CXX      ?= g++
HOST_LEVELS   ?= O2 O3
HOST_BACKENDS ?= scalar avx2 avx_vnni avx512f avx512_vnni
HOST_PIN      ?= $(shell command -v taskset >/dev/null 2>&1 && echo taskset -c 2)
HOST_FLAGS_scalar      :=
HOST_FLAGS_avx2        := -mavx2 -DTINYMIND_ENABLE_SIMD_AVX2=1
HOST_FLAGS_avx_vnni    := $(HOST_FLAGS_avx2) -mavxvnni -DTINYMIND_ENABLE_SIMD_AVX_VNNI=1
HOST_FLAGS_avx512f     := $(HOST_FLAGS_avx2) -mavx512f -mavx512bw -mavx512vl -DTINYMIND_ENABLE_SIMD_AVX512F=1
HOST_FLAGS_avx512_vnni := $(HOST_FLAGS_avx512f) -mavx512vnni -DTINYMIND_ENABLE_SIMD_AVX512_VNNI=1
HOST_CPU_scalar        :=
HOST_CPU_avx2          := avx2
HOST_CPU_avx_vnni      := avx2 avx_vnni
HOST_CPU_avx512f       := avx2 avx512f avx512bw avx512vl
HOST_CPU_avx512_vnni   := avx2 avx512f avx512bw avx512vl avx512_vnni
HB := build/host

$(HB):
	mkdir -p $@

# $(1)=level $(2)=backend
define host_rule
$(HB)/tm_host_bench_$(1)_$(2): host/tm_host_bench.cpp fw/backend.hpp Makefile gen/out/mlp_tinymind.hpp gen/out/kws_tinymind.hpp | $(HB)
	$$(HOST_CXX) -std=c++17 -$(1) -Wall -Wextra -Werror $$(HOST_FLAGS_$(2)) -DTINYMIND_ENABLE_QUANTIZATION=1 \
	    -Ifw -Igen/out -I$$(TM)/cpp -I$$(TM)/cpp/include -I$$(TM)/include $$< -o $$@
endef
$(foreach l,$(HOST_LEVELS),$(foreach b,$(HOST_BACKENDS),$(eval $(call host_rule,$(l),$(b)))))

host-bench: $(foreach l,$(HOST_LEVELS),$(foreach b,$(HOST_BACKENDS),$(HB)/tm_host_bench_$(l)_$(b)))
	@mkdir -p results
	@{ echo "cpu=$$(grep -m1 '^model name' /proc/cpuinfo | sed 's/^[^:]*: *//')"; \
	   echo "compiler=$$($(HOST_CXX) --version | head -1)"; } > results/host_x86.txt
	@for l in $(HOST_LEVELS); do for b in $(HOST_BACKENDS); do \
	  case $$b in $(foreach b,$(HOST_BACKENDS),($(b)) need='$(HOST_CPU_$(b))';;) esac; \
	  miss=""; for f in $$need; do grep -qw "$$f" /proc/cpuinfo || miss="$$miss $$f"; done; \
	  if [ -n "$$miss" ]; then r="skipped=no$$(echo $$miss | sed 's/ /,/g;s/^/:/')"; \
	  else r=$$($(HOST_PIN) ./$(HB)/tm_host_bench_$${l}_$$b) || { echo "level=$$l backend=$$b failed: $$r"; exit 1; }; fi; \
	  echo "level=$$l backend=$$b $$r" | tee -a results/host_x86.txt; \
	done; done

models:
	$(PY) -I gen/make_models.py
	$(PY) -I gen/make_trees.py

TFLM_MAKE = $(MAKE) -C $(TFLM) -f tensorflow/lite/micro/tools/make/Makefile \
            TARGET=cortex_m_generic TARGET_ARCH=$(TFLM_ARCH) microlite
TFLM_O3   = BASE_GENDIR=gen_o3 KERNEL_OPTIMIZATION_LEVEL=-O3 \
            THIRD_PARTY_KERNEL_OPTIMIZATION_LEVEL=-O3 CORE_OPTIMIZATION_LEVEL=-O3

deps:
	test -x $(PY) || { python3 -m venv .venv && .venv/bin/pip install -q tensorflow-cpu emlearn scikit-learn numpy; }
	test -d $(TFLM) || { git clone -q https://github.com/tensorflow/tflite-micro.git $(TFLM) && git -C $(TFLM) checkout -q $(TFLM_REV); }
	$(MAKE) tflm-libs

tflm-libs:
	$(TFLM_MAKE) OPTIMIZED_KERNEL_DIR=cmsis_nn BUILD_TYPE=release
	$(TFLM_MAKE) BUILD_TYPE=release
	$(TFLM_MAKE) OPTIMIZED_KERNEL_DIR=cmsis_nn BUILD_TYPE=release_with_logs
	$(TFLM_MAKE) $(TFLM_O3) OPTIMIZED_KERNEL_DIR=cmsis_nn BUILD_TYPE=release
	$(TFLM_MAKE) $(TFLM_O3) BUILD_TYPE=release
	$(TFLM_MAKE) $(TFLM_O3) OPTIMIZED_KERNEL_DIR=cmsis_nn BUILD_TYPE=release_with_logs

clean:
	rm -rf build

.PHONY: all run run1 probe fit stlink stlink-check clean host-check host-bench models deps tflm-libs
