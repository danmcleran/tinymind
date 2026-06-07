---
title: Examples
layout: default
nav_order: 3
has_children: true
---

# Examples

Every runnable example in [`examples/`](https://github.com/danmcleran/tinymind/tree/master/examples)
trains or runs a model entirely in C++, writes a header-row CSV to its
`output/` directory, and ships a `plot.py` that renders the result. The pages in
this section document each example: what it does, how it works, the exact build
and run commands, and its graphical output.

Reproduce any plot with:

```bash
cd examples/<name>
make release
make run            # writes output/*.csv
make plot           # writes output/*.png  (needs matplotlib in a venv/pyenv)
```

The plot scripts share one style module,
[`examples/plotting/tinymind_plot.py`](https://github.com/danmcleran/tinymind/blob/master/examples/plotting/tinymind_plot.py)
(matplotlib only, headless-safe). The CSV-first contract means you can also drop
the data into pandas or a spreadsheet and build your own visualizations. The
plots on these pages use the dark theme to match this site; `make plot` defaults
to a light theme — set `TINYMIND_PLOT_THEME=dark` to reproduce them exactly.

See the [Example Gallery]({{ site.baseurl }}/gallery) for every plot at a glance,
or pick an example from the navigation. Dataset-driven examples are surveyed in
the [UCI Dataset Capability Report]({{ site.baseurl }}/uci_dataset_capability_report).
