# fuzz

libFuzzer harnesses with a committed corpus. Repo-wide rules are in the root
`AGENTS.md`.

- The `fuzz/` harnesses state properties with `TINYMIND_FUZZ_CHECK` from `fuzz/fuzz_check.hpp` (prints, then traps — never `assert`, which `NDEBUG` strips, and never `return -1`, which libFuzzer reads as "skip"); a harness that only feeds the kernel proves no-crash and nothing more.
- A new byte read in a harness goes **after** every existing read, so the archived crash reproducers in `fuzz/corpus/` still decode as they did.
- `make -C fuzz agree` (HARD) replays the corpus through unsanitized g++/clang++ builds at `-O0`/`-O3` via `fuzz/replay_main.cpp` and requires identical output digests; `make -C fuzz check-coverage` (HARD) requires every property check to execute on the committed corpus.
- `make -C fuzz msan-replay` replays the corpus through `replay_main.cpp` under MemorySanitizer (plain C I/O, so no uninstrumented libstdc++ member runs); `make check-msan` at the top level runs it.
- Run `make check-ci-image` before pushing a change here: the CI compilers are newer than the host's.
