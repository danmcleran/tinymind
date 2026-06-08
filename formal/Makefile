# CBMC bounded-model-checking proofs for the qformat fixed-point kernels.
#
#   make prove      run every proof harness; nonzero exit if any fails
#   make prove-one FUNC=prove_saturate_mul_q8_8
#
# Requires CBMC (sudo apt install cbmc). CBMC ships its own C++ frontend, so no
# compiler/linker is involved -- it analyzes the source directly.
#
# cbmc_qformat.cpp transcribes the qformat.hpp rounding/saturation kernels
# concretely (CBMC's frontend cannot instantiate the templated form). Each
# transcription mirrors the policy arithmetic exactly; see the file header.
CBMC ?= cbmc

SRC = cbmc_qformat.cpp

# Runtime-error checks that constitute the proof obligations. Unwinding is small
# because these kernels are loop-free (shifts, compares, one multiply/divide).
CBMC_CHECKS = --signed-overflow-check --unsigned-overflow-check \
              --conversion-check --div-by-zero-check \
              --pointer-overflow-check --unwind 4 --unwinding-assertions
CBMC_FLAGS  = $(CBMC_CHECKS) --cpp11

FUNCS = prove_saturate_add_sub_int16 \
        prove_saturate_mul_q8_8 \
        prove_saturate_div_q8_8 \
        prove_round_up_q8_8 \
        prove_truncate_q8_8

prove :
	@command -v $(CBMC) >/dev/null 2>&1 || { echo "ERROR: cbmc not found. sudo apt install cbmc"; exit 1; }
	@rc=0; \
	for f in $(FUNCS); do \
		echo "=== CBMC: $$f ==="; \
		$(CBMC) $(CBMC_FLAGS) --function $$f $(SRC) || rc=1; \
	done; \
	if [ $$rc -eq 0 ]; then echo "CBMC: all proofs VERIFIED"; else echo "CBMC: a proof FAILED"; fi; \
	exit $$rc

prove-one :
	@[ -n "$(FUNC)" ] || { echo "usage: make prove-one FUNC=<harness>"; exit 1; }
	$(CBMC) $(CBMC_FLAGS) --function $(FUNC) $(SRC)

.PHONY : prove prove-one
