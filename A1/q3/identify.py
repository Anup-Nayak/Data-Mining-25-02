import sys
import pickle
import networkx as nx

def parse_gspan_fp(fp_path):
    subgraphs = []
    supports = []

    G = None
    support_set = set()

    with open(fp_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # New subgraph
            if line.startswith("t #"):
                if G is not None:
                    subgraphs.append(G)
                    supports.append(support_set)

                G = nx.Graph()
                support_set = set()

            # Node
            elif line.startswith("v"):
                _, nid, label = line.split()
                G.add_node(int(nid), label=int(label))

            # Edge
            elif line.startswith("e"):
                _, u, v, label = line.split()
                G.add_edge(int(u), int(v), label=int(label))

            # Support list
            elif line.startswith("x"):
                parts = line.split()
                for gid in parts[1:]:
                    support_set.add(int(gid))

        # last subgraph
        if G is not None:
            subgraphs.append(G)
            supports.append(support_set)

    return subgraphs, supports

def select_top_k_jaccard(subgraphs, supports, N, k=50, tau=0.8):
    """
    subgraphs : list of nx.Graph
    supports  : list of set(graph_ids)
    N         : number of database graphs
    """

    scored = []
    for g, S in zip(subgraphs, supports):
        c = len(S)
        score = min(c, N - c)
        scored.append((score, g, S))

    scored.sort(reverse=True, key=lambda x: x[0])

    selected = []
    selected_supports = []

    for _, g, S in scored:
        redundant = False
        for S2 in selected_supports:
            jacc = len(S & S2) / len(S | S2)
            if jacc > tau:
                redundant = True
                break

        if not redundant:
            selected.append(g)
            selected_supports.append(S)

        if len(selected) == k:
            break

    return selected


def main():
    if len(sys.argv) != 3:
        print("Usage: python identify.py <gspan_fp_file> <output_features.pkl>")
        sys.exit(1)

    fp_path = sys.argv[1]
    out_path = sys.argv[2]

    print("[1] Parsing gSpan output")
    subgraphs, supports = parse_gspan_fp(fp_path)
    print("    Total subgraphs:", len(subgraphs))

    all_ids = set()
    for S in supports:
        all_ids |= S
    N = max(all_ids) + 1 if all_ids else 0
    print("    Inferred database size:", N)

    print("[2] Support filtering")
    filt_g, filt_s = [], []
    for g, S in zip(subgraphs, supports):
        frac = len(S) / N
        if 0.02 <= frac <= 0.5:
            filt_g.append(g)
            filt_s.append(S)
    print("    After filter:", len(filt_g))

    print("[3] Selecting top-50 (Jaccard)")
    selected = select_top_k_jaccard(
        filt_g, filt_s, N,
        k=50, tau=0.8
    )
    print("    Selected:", len(selected))

    print("[4] Saving features")
    with open(out_path, "wb") as f:
        pickle.dump(selected, f)

    print("Saved to:", out_path)

if __name__ == "__main__":
    main()
