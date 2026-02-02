import sys
import pickle
import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
from joblib import Parallel, delayed
import multiprocessing

def compute_row(G, features):
    row = np.zeros(len(features), dtype=np.uint8)

    node_match = isomorphism.categorical_node_match("label", None)
    edge_match = isomorphism.categorical_edge_match("label", None)

    for j, f in enumerate(features):
        matcher = isomorphism.GraphMatcher(
            G, f,
            node_match=node_match,
            edge_match=edge_match
        )
        if matcher.subgraph_is_isomorphic():
            row[j] = 1

    return row

def load_graphs(dataset_path):
    graphs = []
    G = None

    with open(dataset_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                if G is not None:
                    graphs.append(G)
                G = nx.Graph()

            elif line.startswith("v"):
                _, node_id, label = line.split()
                G.add_node(int(node_id), label=int(label))

            elif line.startswith("e"):
                _, u, v, label = line.split()
                G.add_edge(int(u), int(v), label=int(label))

        if G is not None:
            graphs.append(G)

    return graphs

def build_feature_matrix(graphs, features):
    n_jobs = multiprocessing.cpu_count()

    rows = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(compute_row)(G, features)
        for G in graphs
    )


    X = np.vstack(rows)
    return X

def main():
    graph_path = sys.argv[1]
    feature_path = sys.argv[2]
    out_path = sys.argv[3]

    graphs = load_graphs(graph_path)

    with open(feature_path, "rb") as f:
        features = pickle.load(f)

    X = build_feature_matrix(graphs, features)

    np.save(out_path, X)

if __name__ == "__main__":
    main()
