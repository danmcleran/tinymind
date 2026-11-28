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
"""Check that the four package manifests state one version, and that it is not
behind the newest release tag.

  CMakeLists.txt       project(tinymind VERSION x.y.z)
  library.json         "version": "x.y.z"          (PlatformIO)
  library.properties   version=x.y.z               (Arduino)
  idf_component.yml    version: "x.y.z"            (ESP-IDF)

A release vX.Y sets all four to X.Y.0 before the tag is made. With --tag vX.Y
the version must be exactly X.Y.0; otherwise the newest vX.Y tag in the
repository (if git and the tags are available) must not be newer than it.
Standard library only.
"""

import json
import os
import re
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def read(name):
    with open(os.path.join(REPO, name)) as f:
        return f.read()


def versions():
    found = {}
    m = re.search(r"project\(tinymind\s+VERSION\s+([0-9.]+)", read("CMakeLists.txt"))
    found["CMakeLists.txt"] = m.group(1) if m else None
    found["library.json"] = json.loads(read("library.json")).get("version")
    m = re.search(r"^version=(\S+)$", read("library.properties"), re.M)
    found["library.properties"] = m.group(1) if m else None
    m = re.search(r"^version:\s*\"?([0-9.]+)\"?\s*$", read("idf_component.yml"), re.M)
    found["idf_component.yml"] = m.group(1) if m else None
    return found


def parse(v):
    return tuple(int(x) for x in v.split("."))


def newest_tag():
    try:
        out = subprocess.run(["git", "-C", REPO, "tag", "--list", "v*"], capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    tags = [t for t in out.split() if re.fullmatch(r"v\d+\.\d+", t)]
    return max(tags, key=lambda t: parse(t[1:] + ".0")) if tags else None


def main():
    found = versions()
    for name, v in found.items():
        print(f"{name:20s} {v}")
    bad = [n for n, v in found.items() if v is None or not re.fullmatch(r"\d+\.\d+\.\d+", v)]
    if bad:
        print(f"no x.y.z version found in: {', '.join(bad)}")
        return 1
    if len(set(found.values())) != 1:
        print("the manifests disagree on the version")
        return 1
    version = next(iter(found.values()))
    if "--tag" in sys.argv[1:]:
        tag = sys.argv[sys.argv.index("--tag") + 1]
        want = tag.lstrip("v") + ".0"
        if version != want:
            print(f"release {tag} needs version {want}, the manifests say {version}")
            return 1
    else:
        tag = newest_tag()
        if tag and parse(version) < parse(tag[1:] + ".0"):
            print(f"the manifests say {version}, behind the newest release {tag}")
            return 1
        print(f"newest release tag: {tag or 'none found'}")
    print(f"OK: every manifest says {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
