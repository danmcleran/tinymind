#!/usr/bin/env python3
# Copyright (c) 2026 Dan McLeran
#
"""Render docs/releases.md from the repository's GitHub Releases.

GitHub Releases are the single source of truth for release notes. The Pages
build runs this script before Jekyll so the site's Releases page always
matches them -- including notes edited after a release was published --
instead of carrying a hand-maintained copy that drifts.

Each release becomes an H2 (its name, prefixed with the tag when the name
does not already contain it) with its publish date, followed by its notes. A leading heading that repeats the
release name is dropped, and the remaining headings are demoted so they nest
under the release's H2. Draft releases are skipped. The whole notes block is
wrapped in {% raw %} so Liquid syntax in a release body cannot break the build.

Exits non-zero if no releases are found, so an API failure cannot publish an
empty page.

Usage: python tools/render_releases_page.py [owner/repo] [out.md]
Set GITHUB_TOKEN to authenticate (optional locally; avoids the anonymous rate
limit in CI).
"""
import json
import os
import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

repo = sys.argv[1] if len(sys.argv) > 1 else "danmcleran/tinymind"
out_path = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/releases.md")

HEADING = re.compile(r"^(#{1,6})(\s+.*)$")
FENCE = re.compile(r"^\s*(```|~~~)")


def fetch_releases():
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "tinymind-pages"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    releases = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{repo}/releases?per_page=100&page={page}"
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as resp:
            batch = json.load(resp)
        if not batch:
            return releases
        releases.extend(batch)
        page += 1


def format_body(body, title, tag):
    lines = (body or "").replace("\r\n", "\n").strip().split("\n")

    # The notes usually open with their own "## vX.Y — title" line, which
    # would repeat the H2 this page already emits for the release.
    if lines and HEADING.match(lines[0]) and (tag in lines[0] or title in lines[0]):
        lines = lines[1:]

    # Demote headings so the shallowest one in the notes becomes an H3.
    in_fence = False
    levels = []
    for line in lines:
        if FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence and HEADING.match(line):
            levels.append(len(HEADING.match(line).group(1)))
    shift = max(0, 3 - min(levels)) if levels else 0

    in_fence = False
    out = []
    for line in lines:
        m = None if in_fence else HEADING.match(line)
        if FENCE.match(line):
            in_fence = not in_fence
        if m:
            line = "#" * min(6, len(m.group(1)) + shift) + m.group(2)
        out.append(line)
    text = "\n".join(out).strip()
    if "{% endraw %}" in text:
        sys.exit(f"release {tag}: notes contain '{{% endraw %}}', which would break the raw block")
    return text


def main():
    releases = [r for r in fetch_releases() if not r.get("draft")]
    if not releases:
        sys.exit(f"no published releases found for {repo}; refusing to write an empty page")
    releases.sort(key=lambda r: r.get("published_at") or "", reverse=True)

    parts = [
        "---",
        "title: Releases",
        "layout: default",
        "nav_order: 13",
        "---",
        "",
        "# Releases",
        "",
        "Every TinyMind release, newest first. This page is generated from "
        f"[GitHub Releases](https://github.com/{repo}/releases) each time the "
        "site is built; each release there also carries the source archives.",
        "",
    ]
    for r in releases:
        tag = r["tag_name"]
        name = (r.get("name") or "").strip()
        # Older releases are named after a commit subject; lead with the tag
        # so every heading says which version it is.
        title = name if tag in name else (f"{tag} — {name}" if name else tag)
        date = datetime.strptime(r["published_at"], "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")
        meta = f"Released {date}"
        if r.get("prerelease"):
            meta += " (pre-release)"
        meta += f" · [{tag} on GitHub]({r['html_url']})"
        body = format_body(r.get("body"), title, tag)
        if body == name:
            body = ""  # one-line notes that only repeat the name add nothing
        parts += ["", f"## {title}", "", meta, ""]
        if body:
            parts += ["{% raw %}", body, "{% endraw %}", ""]
        else:
            parts += [""]

    out_path.write_text("\n".join(parts).rstrip("\n") + "\n", encoding="utf-8")
    print(f"Wrote {len(releases)} releases to {out_path}")


main()
