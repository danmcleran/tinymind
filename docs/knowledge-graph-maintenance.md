---
title: Maintenance
layout: default
parent: Knowledge Graph
nav_order: 1
---

# Knowledge Graph Maintenance
{: .no_toc }

How the published graph stays in sync with the code, what is and isn't
versioned, and how to refresh it. For what the graph *is*, see the
[Knowledge Graph]({{ site.baseurl }}/knowledge-graph) page.

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## In sync by construction

The graph is a **derived artifact, regenerated from the source on every GitHub
Pages deploy** — so the published graph always matches `master` and cannot
drift from the code. There is no per-commit "update the graph" step.

The regeneration runs in `.github/workflows/pages.yml`, before the Jekyll
build, and needs **no LLM and no API key**:

1. `pip install graphifyy==<pinned>`
2. `graphify.watch._rebuild_code(Path('.'), force=True)` — re-extracts the
   graph from the current source into `graphify-out/graph.json`
   (`PYTHONHASHSEED=0` pins community detection so the result is
   deterministic).
3. `python tools/render_graph_html.py` — renders the readable
   community-aggregated `graph.html` (see the hairball note below).
4. Both files are copied into `docs/assets/knowledge-graph/` and published.

## What is versioned

Only the **curated community labels** are tracked:

| Path | Tracked? | Why |
|------|----------|-----|
| `graphify-out/.graphify_labels.json` | **yes** | Human/LLM-assigned community names (e.g. "Sigmoid LUT Sizing"). The rebuild reads them; unlabeled communities fall back to "Community N". |
| `graphify-out/graph.json` | no (gitignored) | Regenerated each deploy. |
| `graphify-out/graph.html` | no (gitignored) | Regenerated each deploy. |
| `graphify-out/{GRAPH_REPORT.md,manifest.json}` | no (gitignored) | Regenerated each deploy. |

Do **not** commit the regenerated artifacts — they would churn on every commit.

## Refreshing the community labels

The labels are the one thing a human curates. After you add code that forms
**new clusters**, name them and commit only the labels:

```bash
/graphify .                          # rebuild + (re)label via the graphify skill
git add graphify-out/.graphify_labels.json
git commit -m "graph: label new communities"
```

The next Pages deploy picks up the new names automatically.

## Exploring locally

```bash
/graphify .            # full rebuild
/graphify . --update   # re-extract only changed files
/graphify query "How does the int8 attention path work?"
/graphify path "A" "B" # shortest path between two nodes
```

## The hairball trap

`graphify.watch._rebuild_code` only ever renders the **full node-link graph**.
At ~10k nodes that is an unreadable hairball, and above graphify's viz node
limit it skips `graph.html` entirely. The readable view — one sized bubble per
community — is produced by `export.to_html(..., node_limit=N)` with the graph
*exceeding* `N`, which makes it build a community meta-graph. The interactive
`/graphify` skill does this; `tools/render_graph_html.py` reproduces it
headlessly for CI.

So: **do not** "fix" a skipped `graph.html` by raising `GRAPHIFY_VIZ_NODE_LIMIT`
— that ships the hairball. Run `tools/render_graph_html.py` instead.
