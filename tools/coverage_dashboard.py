#!/usr/bin/env python3
# Copyright (c) 2026 Dan McLeran
#
# Render an LCOV .info file as a single standalone HTML coverage dashboard:
# an overall donut gauge plus per-file horizontal bars, color-graded by
# coverage. No external dependencies, no JavaScript -- pure inline CSS so the
# page works offline straight from disk.
#
# Usage: coverage_dashboard.py <input.info> <output.html>

import html
import sys


def parse_info(path):
    """Return (files, total_hit, total_lines).

    files is a list of dicts: {name, hit, total, pct, uncovered:[lines]}.
    Line counts come from DA: records (DA:<line>,<exec_count>).
    """
    files = []
    name = None
    hit = total = 0
    uncovered = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("SF:"):
                name = line[3:]
                hit = total = 0
                uncovered = []
            elif line.startswith("DA:"):
                body = line[3:]
                parts = body.split(",")
                if len(parts) < 2:
                    continue
                try:
                    lineno = int(parts[0])
                    count = int(parts[1])
                except ValueError:
                    continue
                total += 1
                if count > 0:
                    hit += 1
                else:
                    uncovered.append(lineno)
            elif line == "end_of_record" and name is not None:
                # Trim absolute prefix down to the repo-relative path.
                short = name
                idx = short.rfind("/cpp/")
                if idx != -1:
                    short = short[idx + 1:]
                pct = (100.0 * hit / total) if total else 100.0
                files.append({
                    "name": short,
                    "hit": hit,
                    "total": total,
                    "pct": pct,
                    "uncovered": uncovered,
                })
                name = None

    total_hit = sum(f["hit"] for f in files)
    total_lines = sum(f["total"] for f in files)
    return files, total_hit, total_lines


def grade(pct):
    """Map a percentage to (css_color, css_class)."""
    if pct >= 95.0:
        return "#3fb950"   # green
    if pct >= 90.0:
        return "#9ace3f"   # lime
    if pct >= 75.0:
        return "#d4a017"   # amber
    return "#e5534b"       # red


def render(files, total_hit, total_lines, out_path):
    overall = (100.0 * total_hit / total_lines) if total_lines else 100.0
    overall_color = grade(overall)

    # Worst coverage first so gaps are immediately visible; ties broken by name.
    files_sorted = sorted(files, key=lambda f: (f["pct"], f["name"]))

    rows = []
    for f in files_sorted:
        color = grade(f["pct"])
        pct_txt = "{:.1f}%".format(f["pct"])
        bar_w = "{:.4f}%".format(f["pct"])
        title = ""
        if f["uncovered"]:
            shown = ", ".join(str(n) for n in f["uncovered"][:40])
            if len(f["uncovered"]) > 40:
                shown += ", ..."
            title = " title=\"uncovered lines: {}\"".format(html.escape(shown))
        rows.append(
            "<div class=\"row\"{title}>"
            "<div class=\"name\">{name}</div>"
            "<div class=\"track\"><div class=\"fill\" "
            "style=\"width:{bar};background:{color}\"></div></div>"
            "<div class=\"pct\" style=\"color:{color}\">{pct}</div>"
            "<div class=\"frac\">{hit}/{total}</div>"
            "</div>".format(
                title=title,
                name=html.escape(f["name"]),
                bar=bar_w,
                color=color,
                pct=pct_txt,
                hit=f["hit"],
                total=f["total"],
            )
        )

    full = sum(1 for f in files if f["pct"] >= 99.999)
    file_count = len(files)

    page = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TinyMind Coverage</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 2.5rem 1rem 4rem;
    background: #0d1117; color: #e6edf3;
    font: 15px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}
  .wrap {{ max-width: 920px; margin: 0 auto; }}
  h1 {{ font-size: 1.4rem; font-weight: 600; margin: 0 0 1.5rem; letter-spacing: .2px; }}
  h1 span {{ color: #7d8590; font-weight: 400; }}
  .summary {{
    display: flex; align-items: center; gap: 2rem; flex-wrap: wrap;
    background: #161b22; border: 1px solid #30363d; border-radius: 14px;
    padding: 1.75rem 2rem; margin-bottom: 2rem;
  }}
  .donut {{
    width: 150px; height: 150px; border-radius: 50%; flex: 0 0 auto;
    background: conic-gradient({color} {deg}deg, #21262d {deg}deg 360deg);
    display: grid; place-items: center; position: relative;
  }}
  .donut::after {{
    content: ""; position: absolute; width: 112px; height: 112px;
    border-radius: 50%; background: #161b22;
  }}
  .donut .val {{ position: relative; z-index: 1; text-align: center; }}
  .donut .val b {{ font-size: 1.9rem; font-weight: 700; color: {color}; }}
  .donut .val small {{ display: block; color: #7d8590; font-size: .72rem; }}
  .stats {{ display: flex; gap: 2.25rem; flex-wrap: wrap; }}
  .stat b {{ display: block; font-size: 1.5rem; font-weight: 700; }}
  .stat small {{ color: #7d8590; }}
  .files {{
    background: #161b22; border: 1px solid #30363d; border-radius: 14px;
    padding: .5rem 1.25rem;
  }}
  .row {{
    display: grid; grid-template-columns: 220px 1fr 64px 92px;
    align-items: center; gap: 1rem; padding: .55rem 0;
    border-bottom: 1px solid #21262d;
  }}
  .row:last-child {{ border-bottom: 0; }}
  .name {{ font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: .82rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .track {{ background: #21262d; border-radius: 5px; height: 11px; overflow: hidden; }}
  .fill {{ height: 100%; border-radius: 5px; }}
  .pct {{ text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; font-size: .85rem; }}
  .frac {{ text-align: right; color: #7d8590; font-variant-numeric: tabular-nums; font-size: .78rem; }}
  .legend {{ margin-top: 1.25rem; color: #7d8590; font-size: .78rem; display: flex; gap: 1.25rem; flex-wrap: wrap; }}
  .legend i {{ display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: .4rem; vertical-align: -1px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>TinyMind <span>cpp/ line coverage</span></h1>
  <div class="summary">
    <div class="donut"><div class="val"><b>{overall:.1f}%</b><small>covered</small></div></div>
    <div class="stats">
      <div class="stat"><b>{hit}</b><small>lines covered</small></div>
      <div class="stat"><b>{total}</b><small>lines total</small></div>
      <div class="stat"><b>{miss}</b><small>uncovered</small></div>
      <div class="stat"><b>{full}/{fcount}</b><small>files at 100%</small></div>
    </div>
  </div>
  <div class="files">
{rows}
  </div>
  <div class="legend">
    <span><i style="background:#3fb950"></i>&ge; 95%</span>
    <span><i style="background:#9ace3f"></i>90&ndash;95%</span>
    <span><i style="background:#d4a017"></i>75&ndash;90%</span>
    <span><i style="background:#e5534b"></i>&lt; 75%</span>
    <span>hover a row for uncovered line numbers</span>
  </div>
</div>
</body>
</html>
""".format(
        color=overall_color,
        deg=overall * 3.6,
        overall=overall,
        hit=total_hit,
        total=total_lines,
        miss=total_lines - total_hit,
        full=full,
        fcount=file_count,
        rows="\n".join(rows),
    )

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(page)

    return overall


def main(argv):
    if len(argv) != 3:
        sys.stderr.write("usage: coverage_dashboard.py <input.info> <output.html>\n")
        return 2
    files, total_hit, total_lines = parse_info(argv[1])
    if not files:
        sys.stderr.write("no coverage records found in {}\n".format(argv[1]))
        return 1
    overall = render(files, total_hit, total_lines, argv[2])
    sys.stdout.write("coverage dashboard: {} ({:.1f}% of {} lines)\n".format(
        argv[2], overall, total_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
