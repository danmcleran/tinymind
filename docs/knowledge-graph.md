---
title: Knowledge Graph
layout: default
nav_order: 12
---

# Repository Knowledge Graph
{: .fs-9 }

An automatically-extracted map of how TinyMind's code, docs, and tests connect — built with [graphify](https://github.com/safishamsi/graphify).
{: .fs-6 .fw-300 }

[Open the interactive graph]({{ site.baseurl }}/assets/knowledge-graph/graph.html){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[Raw graph.json](https://github.com/danmcleran/tinymind/blob/master/graphify-out/graph.json){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## What this is

graphify read the entire repository — every `cpp/` header, every example, every doc and test — and turned it into a **knowledge graph**: a network of *nodes* (functions, classes, template types, files, concepts) joined by *edges* (calls, imports, references, conceptual links). It then ran community detection to find the natural module boundaries, with no help from the folder layout.

The result is a navigable picture of the codebase's actual structure — how the pieces depend on one another — rather than how the directories happen to be arranged.

| Metric | Value |
|:-------|:------|
| Files scanned | 373 (225 code · 118 docs · 30 images) |
| Nodes | 9,156 |
| Edges | 12,432 |
| Communities | 1,228 |

Two extraction passes feed the graph: a **structural** pass (deterministic AST parsing of the C++ headers — 8,773 nodes) and a **semantic** pass (LLM reading the docs and example write-ups for named concepts and cross-references — 383 nodes).

## How to read it

Open the [interactive graph]({{ site.baseurl }}/assets/knowledge-graph/graph.html). Because the full graph has more than 5,000 nodes, the view is **aggregated to communities**: each bubble is a cluster, sized by how many nodes it holds, and the lines between bubbles are cross-community links.

**A community is a clump of things that mostly connect to each other.** graphify draws a circle around densely-wired groups — more links *inside* the circle than crossing out of it. Nobody labels these by hand; they fall out of the link structure. For example, `cpp/sigmoid.hpp`'s 246 table-sizing templates reference each other and almost nothing else, so they form one community ("Sigmoid LUT Sizing").

The largest, hand-named communities map cleanly onto the library's real subsystems:

- **Activation LUT families** — separate communities for cos / sin / exp / log / sigmoid / tanh table *sizing* (`cpp/*.hpp`) and *value* tables (8-bit, 32-bit, 128-bit).
- **Neural network core** — gradient & weight updaters, the neuron connection model, layer feed-forward, node-delta / Adam backprop, output/classifier layer (`cpp/neuralnet.hpp`).
- **Int8 quantization family** — `QConv2D`, `QPointwiseConv2D`, `QLSTM`, `QCfC`, `QLayerNorm`, `QAttention Softmax`, affine calibration params.
- **Continuous-time & KAN** — self-attention, reverse-mode autodiff (`revdual.hpp`), KAN backprop policy and B-spline tests.
- **Tooling & tests** — NN / quantization / KAN / LUT-accuracy unit-test suites, the PyTorch importer.

Communities labeled `Community N` (an integer) are real clusters that simply weren't given a topic name — most are tiny 1–4 node fragments in the long tail.

## What the graph revealed

Tracing the graph surfaced one finding worth keeping:

**The Q-format / neural-network pipeline and the int8 affine quantization pipeline are fully independent subgraphs.** They share exactly one seam — `cpp/qbridge.hpp` — which imports `qformat.hpp` on one side and `qaffine.hpp` on the other. Remove that single bridge header and the two worlds fall into disconnected components. This confirms the architectural design rule stated throughout the docs: *"None of the existing layers, `QValue`, or `NeuralNet<>` change; quantized models are built from a parallel layer family."*

{: .note }
> **A caveat when reading the graph.** Generic template type names (`ValueType`, `size_t`, `InputType`) appear once per file in the AST pass, but the connectivity report groups them by *label*. That makes a shared typedef like `ValueType` look like a single high-degree hub spanning many communities, when it is really 58 separate per-file typedefs with no edges between them. Treat any "most connected" claim on a generic type name skeptically — the node's source file disambiguates it.

## Regenerating

The graph is checked in under [`graphify-out/`](https://github.com/danmcleran/tinymind/tree/master/graphify-out). To rebuild it after code changes:

```bash
/graphify .            # full rebuild
/graphify . --update   # re-extract only changed files (uses the cached pass)
/graphify query "How does the int8 attention path work?"
```
