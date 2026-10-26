#!/usr/bin/env bash
# Copyright (c) 2026 Dan McLeran
#
# Run a command inside the toolchain of the GitHub-hosted CI runner image.
#
# CI runs on ubuntu-latest, and the compilers and analyzers it uses are that
# Ubuntu release's packages. When GitHub moves ubuntu-latest to a new release
# (24.04 -> 26.04 brought GCC 15 and cppcheck 2.19), code that builds cleanly
# on a developer machine can fail in CI. This script builds a container image
# from tools/ci-image/Containerfile with the same packages and runs the given
# command in it, so those failures show up before a push.
#
# The working tree (tracked files plus untracked files that are not ignored,
# uncommitted edits included) is copied into the container, so nothing it
# builds lands in your checkout and no host build output leaks in. The copy
# is discarded when the command exits.
#
# Usage:
#   tools/ci-image.sh make check cppcheck
#   tools/ci-image.sh make -C fuzz agree FUZZ_CXX=clang++
#   tools/ci-image.sh bash                    # interactive shell
#   CI_IMAGE_BASE=docker.io/library/ubuntu:24.04 tools/ci-image.sh make check
#
# Needs podman (rootless is fine). The image is rebuilt only when the base or
# the Containerfile changes.

set -euo pipefail

CI_IMAGE_BASE="${CI_IMAGE_BASE:-docker.io/library/ubuntu:26.04}"

repo="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
containerfile="$repo/tools/ci-image/Containerfile"

command -v podman >/dev/null 2>&1 || { echo "ci-image: podman not found (sudo apt install podman)" >&2; exit 1; }
[ "$#" -gt 0 ] || { echo "usage: $0 <command> [args...]" >&2; exit 2; }

hash="$( { echo "$CI_IMAGE_BASE"; cat "$containerfile"; } | sha256sum | cut -c1-12)"
image="localhost/tinymind-ci:$hash"

if ! podman image exists "$image"; then
    echo "ci-image: building $image from $CI_IMAGE_BASE" >&2
    podman build --pull=newer --build-arg BASE="$CI_IMAGE_BASE" \
        -t "$image" -f "$containerfile" "$repo/tools/ci-image" >&2
fi

# The tree goes in with podman cp rather than a bind mount, so the
# container's builds never write into the checkout.
interactive=()
[ -t 0 ] && [ -t 1 ] && interactive=(-i -t)
cid="$(podman create "${interactive[@]}" "$image" "$@")"
trap 'podman rm -f "$cid" >/dev/null 2>&1' EXIT

cd "$repo"
git ls-files -z --cached --others --exclude-standard \
    | tar --null --files-from=- --ignore-failed-read -cf - 2>/dev/null \
    | podman cp - "$cid:/work"

podman start -a "${interactive[@]:0:1}" "$cid"
