# Formal proofs (CBMC) for the int8 requantizer in cpp/qaffine.hpp: free of
# undefined behaviour, and equal to gemmlowp's two-step definition computed
# exactly, over every input of its contract. See README.md.
#
#   make -j16       every proof, then confirm every planted defect is caught
#   make prove      proofs only
#   make mutants    planted defects only (each must be reported)
#
# Needs cbmc and Bitwuzla on PATH (see ../arm_dsp_dot/README.md):
#   PATH=$PWD/../arm_dsp_dot/tools:$PWD/../arm_dsp_dot/.venv/bin:$PATH make -j16

CBMC   ?= cbmc
SOLVER ?= --bitwuzla
ROOT   := ../../../cpp
OUT    := output

CHECKS := --signed-overflow-check --undefined-shift-check
DEPS   := harness.cpp select_props.sh $(ROOT)/qaffine.hpp $(ROOT)/include/tinymind_platform.hpp

# Requantizer's contract is -31 <= shift <= 31. One job per shift value: with
# the shift a constant the solver faces only the multiply.
SHIFTS := $(shell seq -31 31)
APPLY_TYPES := int8_t uint8_t int16_t int32_t

PROOFS := kernels lshift \
          $(foreach t,$(APPLY_TYPES),$(foreach s,$(SHIFTS),apply_$(t)_$(s))) \
          $(foreach s,$(SHIFTS),mbqm_$(s))

# Planted defects, each run at one point where it must be reported:
#   name := sed expression ; property ; dst type ; shift
MUTANTS := threshold nudge rail int32_zp left_limit
MUT_threshold  := s,(mask >> 1) + ((x < 0) ? 1 : 0),(mask >> 1),
MUT_threshold_ARGS := APPLY int8_t 31
MUT_nudge      := s,| static_cast<uint32_t>(0x40000000u);,| static_cast<uint32_t>(0x20000000u);,
MUT_nudge_ARGS := APPLY int8_t 7
# A rail inside int16 range lets the clamped value land inside [qmin, qmax];
# for int8 a 2^14 rail would still be outside, so it is checked at int16.
MUT_rail       := s,const int32_t rail = static_cast<int32_t>(1) << 30;,const int32_t rail = static_cast<int32_t>(1) << 14;,
MUT_rail_ARGS  := APPLY int16_t -3
# The 32-bit-storage fix undone: the zero point is added in int32 again.
MUT_int32_zp   := s,if (sizeof(DstStorageType) >= sizeof(int32_t)),if (false),
MUT_int32_zp_ARGS := APPLY int32_t 7
MUT_left_limit := s,const int32_t limit = INT32_MAX >> left_shift;,const int32_t limit = INT32_MAX >> (left_shift - 1);,
MUT_left_limit_ARGS := MBQM int8_t -5

.PHONY: all prove mutants clean
all: prove mutants

prove: $(addprefix $(OUT)/,$(addsuffix .proved,$(PROOFS)))
	@echo "All $(words $(PROOFS)) requantizer proofs hold."

mutants: $(addprefix $(OUT)/mutant-,$(addsuffix .killed,$(MUTANTS)))
	@echo "All $(words $(MUTANTS)) planted defects were caught."

$(OUT):
	@mkdir -p $@

# $(1) include root, $(2) property, $(3) dst type, $(4) shift ('' for none)
cbmc_cmd = $(CBMC) harness.cpp --cpp11 -I $(1) -D TM_PROP_$(2) -D TM_DST=$(3) \
           $(if $(4),-D TM_SHIFT=$(4)) $(CHECKS)

# A missing function body is a silent unsoundness in CBMC: the call is skipped
# and its outputs become unconstrained. Fail on it rather than trust the run.
define run_proof
	@props=$$(./select_props.sh $(call cbmc_cmd,$(ROOT),$(2),$(3),$(4))); \
	test -n "$$props" || { echo "FAIL   $(1) (no properties selected)"; exit 1; }; \
	$(call cbmc_cmd,$(ROOT),$(2),$(3),$(4)) $$props $(SOLVER) > $(OUT)/$(1).log 2>&1; rc=$$?; \
	if grep -q "no body for function" $(OUT)/$(1).log; then \
	    echo "FAIL   $(1) (missing function model, see $(OUT)/$(1).log)"; exit 1; fi; \
	if [ $$rc -ne 0 ] || ! grep -q "^VERIFICATION SUCCESSFUL" $(OUT)/$(1).log; then \
	    echo "FAIL   $(1) (see $(OUT)/$(1).log)"; exit 1; fi; \
	echo "PROVED $(1)"; touch $(OUT)/$(1).proved
endef

# The two once-only lemmas: every check, no property selection.
$(OUT)/kernels.proved $(OUT)/lshift.proved: $(OUT)/%.proved: $(DEPS) | $(OUT)
	@prop=$$(echo $* | tr a-z A-Z); \
	$(CBMC) harness.cpp --cpp11 -I $(ROOT) -D TM_PROP_$$prop $(CHECKS) $(SOLVER) > $(OUT)/$*.log 2>&1; rc=$$?; \
	if grep -q "no body for function" $(OUT)/$*.log; then \
	    echo "FAIL   $* (missing function model)"; exit 1; fi; \
	if [ $$rc -ne 0 ] || ! grep -q "^VERIFICATION SUCCESSFUL" $(OUT)/$*.log; then \
	    echo "FAIL   $* (see $(OUT)/$*.log)"; exit 1; fi; \
	echo "PROVED $*"; touch $@

# apply_<type>_<shift>: the type name contains an underscore, so split on the
# last one.
$(OUT)/apply_%.proved: $(DEPS) | $(OUT)
	$(call run_proof,apply_$*,APPLY,$(shell echo $* | sed 's/_[^_]*$$//'),$(shell echo $* | sed 's/.*_//'))

$(OUT)/mbqm_%.proved: $(DEPS) | $(OUT)
	$(call run_proof,mbqm_$*,MBQM,int8_t,$*)

# A mutant is caught when CBMC reports any failed property: a wrong result
# (the equivalence assertion) or undefined behaviour. A run that ends in
# neither verdict (a front-end error, a crash) is a failure of the check
# itself, never a catch and never a survivor.
$(OUT)/mutant-%.killed: $(DEPS) | $(OUT)
	@rm -rf $(OUT)/mutant-$*; mkdir -p $(OUT)/mutant-$*; \
	cp -r $(ROOT)/qaffine.hpp $(ROOT)/include $(OUT)/mutant-$*/; \
	sed -i '$(MUT_$*)' $(OUT)/mutant-$*/qaffine.hpp; \
	if cmp -s $(ROOT)/qaffine.hpp $(OUT)/mutant-$*/qaffine.hpp; then \
	    echo "FAIL   mutant $* (sed did not change the header)"; exit 1; fi; \
	set -- $(MUT_$*_ARGS); \
	props=$$(./select_props.sh $(CBMC) harness.cpp --cpp11 -I $(OUT)/mutant-$* -D TM_PROP_$$1 -D TM_DST=$$2 \
	    -D TM_SHIFT=$$3 $(CHECKS)); \
	$(CBMC) harness.cpp --cpp11 -I $(OUT)/mutant-$* -D TM_PROP_$$1 -D TM_DST=$$2 -D TM_SHIFT=$$3 $(CHECKS) \
	    $$props $(SOLVER) > $(OUT)/mutant-$*.log 2>&1; \
	if grep -q "^VERIFICATION FAILED" $(OUT)/mutant-$*.log; then \
	    echo "CAUGHT mutant $* ($$(grep -m1 ': FAILURE' $(OUT)/mutant-$*.log | sed 's/^ *//'))"; touch $@; \
	elif grep -q "^VERIFICATION SUCCESSFUL" $(OUT)/mutant-$*.log; then \
	    echo "FAIL   mutant $* survived (see $(OUT)/mutant-$*.log)"; exit 1; \
	else \
	    echo "FAIL   mutant $*: CBMC reached no verdict (see $(OUT)/mutant-$*.log)"; exit 1; fi

clean:
	rm -rf $(OUT)
