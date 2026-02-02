import sys
import numpy as np

def main():
    db_feat_path = sys.argv[1]
    query_feat_path = sys.argv[2]
    out_path = sys.argv[3]

    DB = np.load(db_feat_path)
    Q = np.load(query_feat_path)

    N_db = DB.shape[0]
    N_q = Q.shape[0]

    with open(out_path, "w") as f:
        for qi in range(N_q):
            vq = Q[qi]

            candidates = []
            for gi in range(N_db):
                vg = DB[gi]
                if np.all(vg >= vq):
                    candidates.append(gi)

            f.write(f"q # {qi}\n")
            if candidates:
                f.write("c # " + " ".join(map(str, candidates)) + "\n")
            else:
                f.write("c #\n")

if __name__ == "__main__":
    main()
