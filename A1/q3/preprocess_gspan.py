import sys
import networkx as nx

inp = sys.argv[1]
out = sys.argv[2]

graphs = []
G = None

with open(inp) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        if line.startswith("#"):
            if G is not None:
                graphs.append(G)
            G = nx.Graph()

        elif line.startswith("v"):
            _, u, l = line.split()
            G.add_node(int(u), label=int(l))

        elif line.startswith("e"):
            _, u, v, l = line.split()
            u, v, l = int(u), int(v), int(l)
            a, b = min(u, v), max(u, v)
            if not G.has_edge(a, b):
                G.add_edge(a, b, label=l)

if G is not None:
    graphs.append(G)

with open(out, "w") as f:
    for i, G in enumerate(graphs):
        f.write(f"t # {i}\n")
        for u in sorted(G.nodes()):
            f.write(f"v {u} {G.nodes[u]['label']}\n")
        for u, v, data in G.edges(data=True):
            f.write(f"e {u} {v} {data['label']}\n")
