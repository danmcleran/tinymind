#!/usr/bin/env python3
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
"""Regenerate the numbers in docs/size-comparison.md.

    python3 tools/size_report/size_report.py instances
        sizeof() of every network and layer configuration the page lists,
        compiled from tools/size_report/instance_sizes.cpp. Takes seconds.

    python3 tools/size_report/size_report.py code [--gcc g++] [--clang clang++]
        GCC vs Clang `.text` (size --format=sysv) for every unit-test binary
        (default build) and every example with a `release` target (-O3).
        Rebuilds each directory with `make clean` first, so it overwrites the
        build products in output/. Takes several minutes.

Both print Markdown tables ready to paste. Standard library only.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Unit-test suites built by their default target. unit_test/embedded is built
# with `all` (the eight gate-matrix smoke binaries). exhaustive and formal are
# left out: exhaustive is a long-running -O2 sweep, formal has no C++ binary.
UNIT_SUITES = ["nn", "qformat", "qformat_oracle", "qlearn", "lookuptable",
               "quantization", "dual", "kan", "pinn", "ltc", "cfc", "integration"]


def run(cmd, cwd, env=None):
    return subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)


def text_size(path):
    out = subprocess.run(["size", "--format=sysv", path], stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True).stdout
    for line in out.splitlines():
        fields = line.split()
        if fields and fields[0] == ".text":
            return int(fields[1])
    return None


def elf_binaries(directory):
    found = []
    if not os.path.isdir(directory):
        return found
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            with open(path, "rb") as handle:
                if handle.read(4) == b"\x7fELF":
                    found.append(path)
    return found


def instances(args):
    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, "instance_sizes")
        cmd = [args.gcc, "-std=c++14", "-Wall", "-Wextra", "-Werror", "-Wpedantic",
               "-DTINYMIND_ENABLE_FLOAT=1", "-DTINYMIND_ENABLE_STD=1",
               "-DTINYMIND_ENABLE_HOSTED_RAND=1",
               "-I" + os.path.join(ROOT, "cpp"), "-I" + os.path.join(ROOT, "cpp", "include"),
               "-I" + os.path.join(ROOT, "include"),
               os.path.join(ROOT, "tools", "size_report", "instance_sizes.cpp"), "-o", exe]
        result = run(cmd, ROOT)
        if result.returncode != 0:
            sys.stderr.write(result.stdout)
            return 1
        values = {}
        for line in subprocess.run([exe], stdout=subprocess.PIPE, text=True).stdout.splitlines():
            key, value = line.split("\t")
            values[key] = int(value)

    def fmt(n):
        return "{:,} bytes".format(n)

    def overhead(train, infer):
        return "+{:,} bytes (+{:.0f}%)".format(train - infer, 100.0 * (train - infer) / infer)

    nets = [("MLP (2->3->1)", "mlp3", 3), ("MLP (2->5->1)", "mlp5", 5),
            ("Elman RNN (2->3->1)", "elman3", 3), ("LSTM (2->3->1)", "lstm3", 3),
            ("GRU (2->3->1)", "gru3", 3), ("KAN (2->5->1, G=5, k=1)", "kan5", 5)]
    for prefix, title in (("double", "`double`"), ("q88", "Q8.8")):
        print("### {}\n".format(title))
        print("| Architecture | Hidden Neurons | Trainable | Non-trainable | Training Overhead |")
        print("|---|---|---|---|---|")
        for label, key, hidden in nets:
            train = values["{}.{}.train".format(prefix, key)]
            infer = values["{}.{}.infer".format(prefix, key)]
            print("| {} | {} | {} | {} | {} |".format(label, hidden, fmt(train), fmt(infer),
                                                    overhead(train, infer)))
        print()

    print("### Liquid cells (parameter counts; bytes = count x sizeof(ValueType))\n")
    for key in ("ltc_2_3", "cfc_2_3_4", "ltc_4_8", "cfc_4_8_16"):
        print("- {}: {} params".format(key, values[key + ".params"]))
    print("\n### Layers\n")
    for key in sorted(values):
        if not key.split(".")[0] in ("double", "q88") and not key.endswith(".params"):
            print("- {}: {}".format(key, fmt(values[key])))
    return 0


def build_and_measure(directory, compiler, target):
    clean = run(["make", "clean"], directory)
    del clean
    cmd = ["make", "CC=" + compiler]
    if target:
        cmd.append(target)
    result = run(cmd, directory)
    if result.returncode != 0:
        return None, result.stdout[-2000:]
    sizes = {}
    for binary in elf_binaries(os.path.join(directory, "output")):
        size = text_size(binary)
        if size is not None:
            sizes[os.path.basename(binary)] = size
    return sizes, None


def code(args):
    rows = {"Unit tests": [], "Examples": []}
    failures = []

    suites = [("unit_test/" + s, None) for s in UNIT_SUITES] + [("unit_test/embedded", "all")]
    examples = []
    for base in ("examples", os.path.join("examples", "pytorch"), os.path.join("examples", "pytorch_quant")):
        for name in sorted(os.listdir(os.path.join(ROOT, base))):
            rel = os.path.join(base, name)
            makefile = os.path.join(ROOT, rel, "Makefile")
            if os.path.isfile(makefile):
                with open(makefile) as handle:
                    if any(line.startswith("release") for line in handle):
                        examples.append((rel, "release"))

    for group, entries in (("Unit tests", suites), ("Examples", examples)):
        for rel, target in entries:
            directory = os.path.join(ROOT, rel)
            measured = {}
            for label, compiler in (("gcc", args.gcc), ("clang", args.clang)):
                sizes, error = build_and_measure(directory, compiler, target)
                if sizes is None:
                    failures.append((rel, label, error))
                    break
                measured[label] = sizes
            run(["make", "clean"], directory)
            if len(measured) != 2:
                continue
            for binary in sorted(set(measured["gcc"]) & set(measured["clang"])):
                prefix = rel.split("/", 1)[1] if group == "Unit tests" else rel.replace("examples/", "", 1)
                name = "{}/{}".format(prefix, binary) if group == "Unit tests" else (
                    prefix if binary == os.path.basename(prefix) else "{} ({})".format(prefix, binary))
                rows[group].append((name, measured["gcc"][binary], measured["clang"][binary]))

    def delta(g, c):
        return 100.0 * (c - g) / g

    print("GCC: " + run([args.gcc, "--version"], ROOT).stdout.splitlines()[0])
    print("Clang: " + run([args.clang, "--version"], ROOT).stdout.splitlines()[0])
    print("\n### Totals\n")
    print("| Group | Binaries | GCC `.text` | Clang `.text` | Delta |")
    print("|---|---|---|---|---|")
    all_g = all_c = count = 0
    for group in ("Unit tests", "Examples"):
        g = sum(r[1] for r in rows[group])
        c = sum(r[2] for r in rows[group])
        all_g += g
        all_c += c
        count += len(rows[group])
        print("| {} | {} | {:,} | {:,} | {:+.1f}% |".format(group, len(rows[group]), g, c, delta(g, c)))
    print("| **All** | **{}** | **{:,}** | **{:,}** | **{:+.1f}%** |".format(count, all_g, all_c, delta(all_g, all_c)))

    smaller = sum(1 for group in rows for r in rows[group] if r[2] < r[1])
    deltas = [delta(r[1], r[2]) for group in rows for r in rows[group]]
    print("\nClang smaller for {} of {}; spread {:+.1f}% to {:+.1f}%".format(
        smaller, count, min(deltas), max(deltas)))

    for group in ("Unit tests", "Examples"):
        print("\n### {}\n".format(group))
        print("| {} | GCC `.text` | Clang `.text` | Delta |".format("Binary" if group == "Unit tests" else "Example"))
        print("|---|---|---|---|")
        for name, g, c in sorted(rows[group], key=lambda r: delta(r[1], r[2])):
            print("| `{}` | {:,} | {:,} | {:+.1f}% |".format(name, g, c, delta(g, c)))

    for rel, label, error in failures:
        sys.stderr.write("BUILD FAILED: {} with {}\n{}\n".format(rel, label, error))
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("instances", "code"))
    parser.add_argument("--gcc", default="g++")
    parser.add_argument("--clang", default="clang++")
    args = parser.parse_args()
    for tool in (args.gcc, "make", "size") + ((args.clang,) if args.mode == "code" else ()):
        if shutil.which(tool) is None:
            sys.stderr.write("missing tool: {}\n".format(tool))
            return 1
    return instances(args) if args.mode == "instances" else code(args)


if __name__ == "__main__":
    sys.exit(main())
