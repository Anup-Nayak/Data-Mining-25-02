import sys

inp = sys.argv[1]
out = sys.argv[2]

# FIXED label ordering (DO NOT CHANGE)
label_order = ['Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S', 'Si']
label_to_int = {label: idx for idx, label in enumerate(label_order)}

# Read entire input first
with open(inp, "r") as f:
    lines = [l.strip() for l in f if l.strip()]

with open(out, "w") as g:
    i = 0
    tid = 1  # transaction counter

    while i < len(lines):
        # Ignore original graph ID
        g.write(f"t # {tid}\n")
        tid += 1
        i += 1  # skip original "#graphID" line

        # number of nodes
        n = int(lines[i])
        i += 1

        # node labels → convert to integers using fixed mapping
        for v in range(n):
            label = lines[i]
            if label not in label_to_int:
                raise ValueError(f"Unknown node label: {label}")
            mapped_label = label_to_int[label]
            g.write(f"v {v} {mapped_label}\n")
            i += 1

        # number of edges
        m = int(lines[i])
        i += 1

        # edges (edge labels left unchanged)
        for _ in range(m):
            u, v, l = lines[i].split()
            g.write(f"e {u} {v} {l}\n")
            i += 1
