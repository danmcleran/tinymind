# Cross-framework NN benchmark on NUCLEO-F411RE (Cortex-M4F @ 100 MHz):
# tinymind vs TensorFlow Lite for Microcontrollers (CMSIS-NN and reference
# kernels) vs emlearn. See README.md for method and results.
#
#   make deps                 venv (TF, emlearn, sklearn) + TFLM clone + TFLM libs
#   make host-check           tinymind int8 outputs reproduce the expected hashes
#   make host-bench           time tinymind on the build host, scalar and each x86
#                             SIMD backend the CPU supports -> results/host_x86.txt
#   make LEVEL=O2|O3          build every image into build/<LEVEL>/
#   make LEVEL=O2|O3 run      flash + capture each image -> results/raw_<LEVEL>.txt
#   make LEVEL=.. IMG=x run1  flash + capture one image
#   make probe                bisect the minimum TFLM tensor arena on the board
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
TFLM_CMSIS := $(TFLM_GEN)/cortex_m_generic_cortex-m4+fp_release_cmsis_nn_gcc/lib/libtensorflow-microlite.a
TFLM_REF   := $(TFLM_GEN)/cortex_m_generic_cortex-m4+fp_release_gcc/lib/libtensorflow-microlite.a
TFLM_LOGS  := $(TFLM_GEN)/cortex_m_generic_cortex-m4+fp_release_with_logs_cmsis_nn_gcc/lib/libtensorflow-microlite.a
EML_INC ?= $(shell $(PY) -I -c "import emlearn; print(emlearn.includedir)" 2>/dev/null)
B       := build/$(LEVEL)

# Which ST-LINK to flash and read. With one probe attached it is found on its
# own; with several, STLINK=<serial> picks one (`make stlink` lists them), on
# the command line, in the environment, or in an untracked local.mk. Both
# openocd and the VCP port follow the serial, so the two cannot disagree and
# replugging (which renumbers /dev/ttyACM*) does not matter.
-include local.mk
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
CPU  = -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16
COMMON = $(CPU) $(OPT) -g -ffunction-sections -fdata-sections -Wall -Wextra -Werror
CXXFLAGS = $(COMMON) -std=c++17 -fno-exceptions -fno-rtti -fno-threadsafe-statics -fno-use-cxa-atexit
CFLAGS   = $(COMMON) -std=c99
INC  = -Iboard -Ifw -Igen/out
LD   = -T board/stm32f411re.ld -nostartfiles --specs=nano.specs --specs=nosys.specs -Wl,--gc-sections
HARNESS = board/board.cpp fw/harness.cpp

TM_INC   = -I$(TM)/cpp -I$(TM)/cpp/include -I$(TM)/include -DTINYMIND_ENABLE_QUANTIZATION=1
DSP      = -DTINYMIND_ENABLE_SIMD_ARM_DSP=1
TFLM_INC = -isystem $(TFLM) -isystem $(TFLM)/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
           -isystem $(TFLM)/tensorflow/lite/micro/tools/make/downloads/gemmlowp \
           -DTF_LITE_STRIP_ERROR_STRINGS -DNDEBUG -funsigned-char

IMAGES = baseline \
  tm_mlp tm_mlp_dsp tm_kws tm_kws_dsp \
  tflm_cmsis_mlp_int8 tflm_ref_mlp_int8 tflm_cmsis_mlp_f32 \
  tflm_cmsis_kws_int8 tflm_ref_kws_int8 tflm_cmsis_kws_f32 \
  eml_mlp
PROF_IMAGES = tm_mlp_prof tm_mlp_dsp_prof tm_kws_prof tm_kws_dsp_prof \
  tflm_cmsis_mlp_int8_prof tflm_cmsis_kws_int8_prof

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

# emlearn's generated MLP ends in softmax; patch it to identity so it returns
# logits like the other backends (argmax is unchanged).
$(B)/mlp_emlearn_logits.h: gen/out/mlp_emlearn.h | $(B)
	sed 's/EmlNetActivationSoftmax/EmlNetActivationIdentity/' $< > $@

$(B)/baseline.elf: fw/be_baseline.cpp $(HARNESS) | $(B)
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $^ $(LD)

# $(1)=image $(2)=extra defines
define tm_rule
$(B)/$(1).elf: fw/be_tinymind.cpp $$(HARNESS) | $$(B)
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
$(B)/$(1).elf: fw/be_tflm.cpp $$(HARNESS) $(2) | $$(B)
	$$(CXX) $$(CXXFLAGS) $$(INC) $$(TFLM_INC) $(3) -DKERNELS_TAG='"$(4)"' -DARENA_BYTES=$$(ARENA_$(1)) -o $$@ fw/be_tflm.cpp $$(HARNESS) $(2) -lm $$(LD)
$(B)/$(1)_prof.elf: fw/be_tflm.cpp $$(HARNESS) $$(TFLM_LOGS) | $$(B)
	$$(CXX) $$(CXXFLAGS) $$(INC) $$(TFLM_INC) -UTF_LITE_STRIP_ERROR_STRINGS -DPROFILE $(3) -DKERNELS_TAG='"$(4)"' -DARENA_BYTES=$$(ARENA_$(1)) -o $$@ fw/be_tflm.cpp $$(HARNESS) $$(TFLM_LOGS) -lm $$(LD)
$(B)/$(1)_probe.elf: fw/be_tflm.cpp $$(HARNESS) $(2) | $$(B)
	$$(CXX) $$(CXXFLAGS) $$(INC) $$(TFLM_INC) -DARENA_PROBE $(3) -DKERNELS_TAG='"$(4)"' -DARENA_BYTES=32768 -o $$@ fw/be_tflm.cpp $$(HARNESS) $(2) -lm $$(LD)
endef
$(eval $(call tflm_rule,tflm_cmsis_mlp_int8,$(TFLM_CMSIS),-DMODEL_MLP,+cmsis))
$(eval $(call tflm_rule,tflm_ref_mlp_int8,$(TFLM_REF),-DMODEL_MLP,+ref))
$(eval $(call tflm_rule,tflm_cmsis_mlp_f32,$(TFLM_CMSIS),-DMODEL_MLP -DDTYPE_F32,+cmsis))
$(eval $(call tflm_rule,tflm_cmsis_kws_int8,$(TFLM_CMSIS),-DMODEL_KWS,+cmsis))
$(eval $(call tflm_rule,tflm_ref_kws_int8,$(TFLM_REF),-DMODEL_KWS,+ref))
$(eval $(call tflm_rule,tflm_cmsis_kws_f32,$(TFLM_CMSIS),-DMODEL_KWS -DDTYPE_F32,+cmsis))

$(B)/eml_mlp.o: fw/be_emlearn.c $(B)/mlp_emlearn_logits.h | $(B)
	@test -n "$(EML_INC)" || { echo "emlearn not found: run 'make deps' or pass EML_INC=<emlearn include dir>"; exit 1; }
	$(CC) $(CFLAGS) $(INC) -I$(B) -isystem $(EML_INC) -c -o $@ $<
$(B)/eml_mlp.elf: $(B)/eml_mlp.o $(HARNESS) | $(B)
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $^ -lm $(LD)

OPENOCD = openocd -f interface/stlink.cfg -c "adapter serial $(STLINK)" -f target/stm32f4x.cfg

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
	  { echo "flashing $(IMG) via ST-LINK $(STLINK) failed: is that probe on the F411?"; exit 1; }
	-fuser -k $(PORT) >/dev/null 2>&1; stty -F $(PORT) 115200 raw -echo
	( timeout 8 cat $(PORT) > $(B)/$(IMG).raw & ) ; sleep 0.5 ; \
	  $(OPENOCD) -c "init; reset run; shutdown" >/dev/null 2>&1 ; sleep 6
	tr -d '\r' < $(B)/$(IMG).raw | sed -n '/^# begin bench/,/^# end bench/p' > $(B)/$(IMG).txt
	@grep -q '^# end bench' $(B)/$(IMG).txt || { echo "capture failed for $(IMG)"; exit 1; }
	@cat $(B)/$(IMG).txt

run: all stlink-check
	@rm -f results/raw_$(LEVEL).txt
	@for i in $(IMAGES) $(PROF_IMAGES); do $(MAKE) -s LEVEL=$(LEVEL) IMG=$$i run1 > /dev/null && \
	  { echo "level=$(LEVEL)"; echo "image=$$i"; cat $(B)/$$i.txt; } >> results/raw_$(LEVEL).txt || exit 1; done
	@for i in $(IMAGES); do $(SIZE) -A $(B)/$$i.elf | awk -v n=$$i '/^\.(isr_vector|text|rodata|ARM|init_array|data|bss)/{s[$$1]+=$$2} END{printf "size image=%s flash=%d ram=%d\n", n, s[".isr_vector"]+s[".text"]+s[".rodata"]+s[".ARM.exidx"]+s[".init_array"]+s[".data"], s[".data"]+s[".bss"]}'; done >> results/raw_$(LEVEL).txt

PROBES = tflm_cmsis_mlp_int8 tflm_ref_mlp_int8 tflm_cmsis_mlp_f32 tflm_cmsis_kws_int8 tflm_ref_kws_int8 tflm_cmsis_kws_f32
probe: $(PROBES:%=$(B)/%_probe.elf) stlink-check
	@for i in $(PROBES); do $(MAKE) -s LEVEL=$(LEVEL) IMG=$${i}_probe run1 | grep -E "^(name|arena_used|setup_rc)"; done

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

TFLM_MAKE = $(MAKE) -C $(TFLM) -f tensorflow/lite/micro/tools/make/Makefile \
            TARGET=cortex_m_generic TARGET_ARCH=cortex-m4+fp microlite
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

.PHONY: all run run1 probe stlink stlink-check clean host-check host-bench models deps tflm-libs
