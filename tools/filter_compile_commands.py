#!/usr/bin/env python3
# Copyright (c) 2026 Dan McLeran
#
# Drop the builds that are meant to fail to compile from a compile_commands.json.
#
# unit_test/embedded's simd_prereq_regressions and qtree_static_regressions
# checks compile planted defects that must be refused, and mark each such build
# with -DTINYMIND_EXPECT_COMPILE_FAILURE (EXPECT_FAIL in that Makefile). bear
# records them like any other compile, so clang-tidy would report the planted
# failures as errors in the library and analyse those files under flags that
# cannot compile. The top-level `tidy` target runs this on the database first.
#
# Usage: filter_compile_commands.py <compile_commands.json>   (rewritten in place)

import json
import sys

MARKER = "TINYMIND_EXPECT_COMPILE_FAILURE"


def is_expected_failure(entry):
    # bear writes "arguments" (a list); other generators write "command".
    text = " ".join(entry.get("arguments", [])) + " " + entry.get("command", "")
    return MARKER in text


def main():
    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as fh:
        db = json.load(fh)
    keep = [e for e in db if not is_expected_failure(e)]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(keep, fh, indent=2)
    print(f"{path}: dropped {len(db) - len(keep)} expected-compile-failure entries, kept {len(keep)}")


if __name__ == "__main__":
    main()
