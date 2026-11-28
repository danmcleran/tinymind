#!/usr/bin/env python3
#
# Copyright (c) 2026 Dan McLeran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""Write (or, with --check, verify) src/: the layout Arduino and PlatformIO
expect of a library.

Arduino's library format puts only src/ on the include path and compiles
only what is under src/. tinymind's headers live in cpp/ and cpp/include/, so
src/ holds:

  * one forwarding header per header directly in cpp/ and cpp/include/, of
    the same name, that includes the real one, so `#include "qdense.hpp"`
    (and every #include a generated model.hpp makes) resolves from src/;
  * tinymind.h, the header Arduino's library.properties names;
  * tinymind_lookup_tables.cpp, which compiles cpp/lookupTables.cpp (an
    empty object unless a TINYMIND_USE_* flag is set).

Run it after adding or removing a header in cpp/ or cpp/include/. The
packaging CI job runs it with --check, which fails if src/ is out of date.
Standard library only.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(REPO, "src")

BANNER = ("/**\n"
          "* Copyright (c) 2026 Dan McLeran\n"
          "*\n"
          "* GENERATED FILE - do not edit by hand. Regenerate via tools/gen_src_forwarders.py.\n"
          "*/\n")

UMBRELLA = BANNER + """
// tinymind for Arduino and PlatformIO. Include this, then the headers you
// use by name (for example "qdense.hpp"): src/ forwards every header in
// cpp/ and cpp/include/. Set the TINYMIND_ENABLE_* feature macros, and any
// TINYMIND_USE_* lookup tables, for the whole build (build flags), so the
// library's sources and yours agree.
#pragma once

#include "../cpp/include/tinymind_platform.hpp"
"""

LOOKUP_TABLES = BANNER + """
// cpp/lookupTables.cpp, compiled as part of the library. It is an empty
// object unless a TINYMIND_USE_<FUNCTION>_<I>_<F> flag is set.
#include "../cpp/lookupTables.cpp"
"""


def expected():
    files = {"tinymind.h": UMBRELLA, "tinymind_lookup_tables.cpp": LOOKUP_TABLES}
    for sub in ("cpp", os.path.join("cpp", "include")):
        for name in sorted(os.listdir(os.path.join(REPO, sub))):
            if not name.endswith(".hpp") or not os.path.isfile(os.path.join(REPO, sub, name)):
                continue
            target = "../" + sub.replace(os.sep, "/") + "/" + name
            if name in files:
                sys.exit(f"two headers named {name}: src/ cannot forward both")
            files[name] = BANNER + f"#pragma once\n#include \"{target}\"\n"
    return files


def main():
    check = "--check" in sys.argv[1:]
    want = expected()
    have = set(os.listdir(SRC)) if os.path.isdir(SRC) else set()
    stale = sorted(have - set(want))
    wrong = []
    for name, text in sorted(want.items()):
        path = os.path.join(SRC, name)
        current = open(path).read() if os.path.exists(path) else None
        if current != text:
            wrong.append(name)
            if not check:
                os.makedirs(SRC, exist_ok=True)
                with open(path, "w") as f:
                    f.write(text)
    if not check:
        for name in stale:
            os.remove(os.path.join(SRC, name))
        print(f"src/: {len(want)} files ({len(wrong)} written, {len(stale)} removed)")
        return 0
    if wrong or stale:
        for name in wrong:
            print(f"src/{name}: missing or out of date")
        for name in stale:
            print(f"src/{name}: forwards no header in cpp/ or cpp/include/")
        print("run tools/gen_src_forwarders.py and commit src/")
        return 1
    print(f"src/: {len(want)} files, up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
