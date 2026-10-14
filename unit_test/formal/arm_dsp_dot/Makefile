# Formal equivalence proof (CBMC): the SMLAD backend of int8DotWithZeroPoint
# returns the same int32 as the scalar reference for every input.
#
#   make            prove every configuration, then confirm every mutant is caught
#   make prove      proofs only
#   make mutants    planted defects only (each must be reported)
#
# Needs cbmc and an SMT solver on PATH. MiniSat (CBMC's built-in SAT backend)
# times out on the multiplier equivalence; a word-level bit-vector solver does
# not. Bitwuzla is fastest; SOLVER=--z3 also works, about 5x slower.

CBMC   ?= cbmc
SOLVER ?= --bitwuzla
ROOT   := ../../../cpp
OUT    := output

COMMON := --cpp11 -I shim -D TINYMIND_ENABLE_SIMD_ARM_DSP=1 -D __ARM_FEATURE_DSP=1
CHECKS := --unwinding-assertions --bounds-check --pointer-check \
          --signed-overflow-check --undefined-shift-check

DEPS := harness.cpp shim/arm_acle.h $(ROOT)/include/simd/simd_arm_dsp.hpp \
        $(ROOT)/include/simd/simd_dispatch.hpp

# Proof configurations, named <endian>-<n>-<xoff>-<woff>. Every length 0..8
# covers zero, one and two SMLAD blocks with each tail length 0..3; the
# offset runs cover misaligned buffers; the big-endian runs check the
# header's claim that its byte pairing does not depend on load endianness.
LENGTHS := 0 1 2 3 4 5 6 7 8
PROOFS  := $(foreach n,$(LENGTHS),little-$(n)-0-0) \
           little-5-1-3 little-7-2-2 little-7-3-1 \
           big-4-0-0 big-7-0-0 big-7-1-3

# Planted defects. Each must make the equivalence assertion fail; a mutant
# that survives means the harness is not checking what it claims to.
MUT_pairing  := s,acc = __smlad(sxtb16Ror8(wv),acc = __smlad(__sxtb16(wv),
MUT_zp_sign  := s,-static_cast<int16_t>(zp),static_cast<int16_t>(zp),
MUT_tail     := s,(n & 3u),(n \& 1u),
# Rotate the weight word the wrong way (first occurrence only, sxtb16Ror8).
# Rotating BOTH words the wrong way would be an equivalent mutant: lanes
# (b3, b1) instead of (b1, b3) on both sides still pair w[i] with x[i].
MUT_rotate   := 0,/(u >> 8) | (u << 24)/s,,(u << 8) | (u >> 24),
MUTANTS := pairing zp_sign tail rotate
MUTANT_N := 7

.PHONY: all prove mutants clean
all: prove mutants

prove: $(addprefix $(OUT)/,$(addsuffix .proved,$(PROOFS)))
	@echo "All $(words $(PROOFS)) proofs hold."

mutants: $(addprefix $(OUT)/mutant-,$(addsuffix .killed,$(MUTANTS)))
	@echo "All $(words $(MUTANTS)) planted defects were caught."

$(OUT):
	@mkdir -p $@

# A missing function body is a silent unsoundness in CBMC: the call is skipped
# and its outputs become unconstrained. Fail on it rather than trust the run.
$(OUT)/%.proved: $(DEPS) | $(OUT)
	@set -- $(subst -, ,$*); \
	$(CBMC) harness.cpp $(COMMON) -I $(ROOT) -D TM_N=$$2 -D TM_XOFF=$$3 -D TM_WOFF=$$4 \
	    --$$1-endian --unwind $$(($$2 + 5)) $(CHECKS) $(SOLVER) > $(OUT)/$*.log 2>&1; rc=$$?; \
	if grep -q "no body for function" $(OUT)/$*.log; then \
	    echo "FAIL   $* (missing function model, see $(OUT)/$*.log)"; exit 1; fi; \
	if [ $$rc -ne 0 ] || ! grep -q "^VERIFICATION SUCCESSFUL" $(OUT)/$*.log; then \
	    echo "FAIL   $* (see $(OUT)/$*.log)"; exit 1; fi; \
	echo "PROVED $*"; touch $@

$(OUT)/mutant-%.killed: $(DEPS) | $(OUT)
	@rm -rf $(OUT)/mutant-$*; mkdir -p $(OUT)/mutant-$*; \
	cp -r $(ROOT)/include $(OUT)/mutant-$*/include; \
	sed -i '$(MUT_$*)' $(OUT)/mutant-$*/include/simd/simd_arm_dsp.hpp; \
	if cmp -s $(ROOT)/include/simd/simd_arm_dsp.hpp $(OUT)/mutant-$*/include/simd/simd_arm_dsp.hpp; then \
	    echo "FAIL   mutant $* (sed did not change the header)"; exit 1; fi; \
	$(CBMC) harness.cpp $(COMMON) -I $(OUT)/mutant-$* -D TM_N=$(MUTANT_N) \
	    --little-endian --unwind $$(($(MUTANT_N) + 5)) --unwinding-assertions \
	    --property main.assertion.1 $(SOLVER) > $(OUT)/mutant-$*.log 2>&1; \
	if grep -q "SMLAD backend equals scalar reference: FAILURE" $(OUT)/mutant-$*.log; then \
	    echo "CAUGHT mutant $*"; touch $@; \
	else \
	    echo "FAIL   mutant $* survived (see $(OUT)/mutant-$*.log)"; exit 1; fi

clean:
	rm -rf $(OUT)
