#!/usr/bin/env python3
"""Render the community-aggregated knowledge-graph HTML from graphify-out/graph.json.

graphify's headless rebuild (graphify.watch._rebuild_code) only ever renders the
full node-link graph -- for a ~10k-node graph that is an unreadable hairball, and
above the viz limit it skips graph.html entirely. The readable view (the one the
Pages docs describe: one sized bubble per community) is produced by calling
export.to_html with an explicit node_limit the graph exceeds, which makes it
build a community meta-graph instead. The interactive `/graphify` skill does that;
this script reproduces it headlessly so the Pages CI can publish the same view
without an LLM.

Usage: python tools/render_graph_html.py [graph.json] [labels.json] [out.html]
"""
import json
import sys
from pathlib import Path

import networkx as nx
from graphify.export import to_html

graph_path  = Path(sys.argv[1] if len(sys.argv) > 1 else "graphify-out/graph.json")
labels_path = Path(sys.argv[2] if len(sys.argv) > 2 else "graphify-out/.graphify_labels.json")
out_path    = Path(sys.argv[3] if len(sys.argv) > 3 else "graphify-out/graph.html")

data = json.loads(graph_path.read_text(encoding="utf-8"))

# Rebuild the graph by hand (version-independent; avoids node_link_graph's
# changing edges-key API). graph.json is node-link with edges under "links".
G = nx.DiGraph() if data.get("directed") else nx.Graph()
for n in data["nodes"]:
    nid = n["id"]
    G.add_node(nid, **{k: v for k, v in n.items() if k != "id"})
for e in data.get("links", []):
    G.add_edge(e["source"], e["target"],
               **{k: v for k, v in e.items() if k not in ("source", "target")})

# communities: {community_id -> [node_id, ...]} from each node's community attr.
communities: dict[int, list[str]] = {}
for nid, attrs in G.nodes(data=True):
    cid = attrs.get("community")
    if cid is None:
        continue
    communities.setdefault(int(cid), []).append(nid)

labels: dict[int, str] = {}
if labels_path.exists():
    labels = {int(k): v for k, v in json.loads(labels_path.read_text(encoding="utf-8")).items()}

member_counts = {cid: len(members) for cid, members in communities.items()}

# node_limit below the node count forces the aggregated community meta-graph;
# member_counts sizes each bubble by how many nodes its community holds.
to_html(G, communities, str(out_path),
        community_labels=labels, member_counts=member_counts, node_limit=5000)

print(f"rendered aggregated graph: {G.number_of_nodes()} nodes / "
      f"{len(communities)} communities -> {out_path}")
