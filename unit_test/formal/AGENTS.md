# unit_test/formal

CBMC proofs. Repo-wide rules are in the root `AGENTS.md`.

`formal/` holds CBMC proofs that need Bitwuzla (pinned by hash in `formal/requirements.txt`): `arm_dsp_dot` (SMLAD dot product ≡ scalar; HARD, CI job "CBMC proof (SMLAD dot product)" via `make ci`, which builds the venv) and `requantizer` (opt-in, ~11 CPU-hours) (`Requantizer::apply` and `multiplyByQuantizedMultiplier` UB-free and equal to gemmlowp's definition for every input; one job per shift; it found the shift > 31 and 32-bit-storage zero-point UB fixed with it). `Requantizer`'s contract is -31 <= shift <= 31, which `quantizeMultiplier` guarantees. CBMC 5.95's C++ front end rejects anonymous namespaces, member function templates and static members of explicit specializations (so `FastRequantizer` is unproven); a mutant check must treat "no verdict" as a failure, never as caught or survived.
