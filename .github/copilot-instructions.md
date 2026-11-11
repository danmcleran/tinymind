# GitHub Copilot Instructions for tinymind

The instructions for every coding agent are in [`AGENTS.md`](../AGENTS.md) at the
repository root, with further `AGENTS.md` files in the directories it lists.
Follow those; this file adds nothing to them.

The rules most often missed:

- C++14 is the language floor for `cpp/`, tests and examples (`make check-cxx14`).
- A fixed-point SGD step is built with `multiplyRoundNearest` (gradient) and `multiplyRoundTowardZero` (momentum), never a raw multiply.
- No heap allocation on the embedded paths; every header includes `include/tinymind_platform.hpp` first and uses `#pragma once`.
- Allman braces; keep each file's line endings.
- Changes reach `master` through a pull request with the required checks green.
