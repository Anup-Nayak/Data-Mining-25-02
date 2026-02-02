import sys

inp = sys.argv[1]
out = sys.argv[2]

# Read entire input first
with open(inp, "r") as f:
    lines = [l.strip() for l in f if l.strip()]

# Now write output
with open(out, "w") as g:
    i = 0
    while i < len(lines):
        # graph ID line: "#graphID"
        graph_id = lines[i].replace("#", "").strip()
        g.write(f"t # {graph_id}\n")
        i += 1

        # number of nodes
        n = int(lines[i])
        i += 1

        # node labels
        for v in range(n):
            label = lines[i]
            g.write(f"v {v} {label}\n")
            i += 1

        # number of edges
        m = int(lines[i])

        i += 1


        # edges
        for _ in range(m):
            u, v, l = lines[i].split(" ")
            g.write(f"u {u.strip()} {v.strip()} {l.strip()}\n")
            i += 1




