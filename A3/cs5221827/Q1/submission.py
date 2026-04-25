import numpy as np


def solve(base_vectors, query_vectors, k, K, time_budget):
    import faiss
    import os
    import time

    t0 = time.perf_counter()

    base = np.ascontiguousarray(base_vectors, dtype=np.float32)
    queries = np.ascontiguousarray(query_vectors, dtype=np.float32)

    N, d = base.shape
    Q = queries.shape[0]

    if N == 0 or K <= 0:
        return np.empty((0,), dtype=np.int64)

    k_eff = int(max(1, min(int(k), N)))
    K_eff = int(max(1, min(int(K), N)))

    fixed_nprobe = True
    nprobe_min = 1
    nprobe_max = 1
    nprobe = 1

    if N <= 300_000:
        index = faiss.IndexFlatL2(d)
        index.add(base)
        fixed_nprobe = True
    else:
        nlist = int(min(8192, max(2048, round(np.sqrt(N) * 4))))
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

        train_size = int(min(N, max(nlist * 40, 100_000)))
        step = max(1, N // train_size)
        train_vecs = np.ascontiguousarray(base[::step][:train_size], dtype=np.float32)
        index.train(train_vecs)
        index.add(base)

        nprobe_override = os.getenv("A3_NPROBE_OVERRIDE")
        if nprobe_override is not None:
            try:
                nprobe = max(1, int(nprobe_override))
                fixed_nprobe = True
            except ValueError:
                fixed_nprobe = False
                nprobe = 1
        else:
            fixed_nprobe = False
            nprobe = 1

        nprobe_min = min(nlist, max(8, nlist // 256))
        nprobe_max = min(nlist, max(64, nlist // 32))
        if fixed_nprobe:
            nprobe = min(nlist, nprobe)
        else:
            budget_factor = float(time_budget) / (float(time_budget) + 10.0)
            nprobe_seed = int(round((0.035 * nlist) * budget_factor))
            nprobe = min(nprobe_max, max(nprobe_min, nprobe_seed))
        index.nprobe = int(nprobe)

    selected_queries = queries

    counts = np.zeros(N, dtype=np.int32)
    batch_size = 8192 if N >= 800_000 else (4096 if N >= 500_000 else 2048)
    deadline = t0 + max(0.01, float(time_budget)) * 0.985
    processed_q = 0
    target_runtime = max(0.01, float(time_budget)) * 0.95
    batch_no = 0

    for start in range(0, selected_queries.shape[0], batch_size):
        if time.perf_counter() >= deadline:
            break

        if N > 300_000 and not fixed_nprobe and processed_q > 0:
            elapsed = time.perf_counter() - t0
            should_adjust = (
                processed_q >= 3 * batch_size and (batch_no % 4 == 0)
            )
            if elapsed > 0 and should_adjust:
                qps = processed_q / elapsed
                rem_q = selected_queries.shape[0] - processed_q
                projected_total = elapsed + (rem_q / max(qps, 1e-9))

                if projected_total > target_runtime * 1.10 and nprobe > nprobe_min:
                    nprobe = max(nprobe_min, int(nprobe * 0.95))
                    index.nprobe = int(nprobe)

        end = min(start + batch_size, selected_queries.shape[0])
        _, I = index.search(selected_queries[start:end], k_eff)
        valid = I[I >= 0]
        if valid.size:
            np.add.at(counts, valid.astype(np.int64, copy=False), 1)
        processed_q = end
        batch_no += 1

    all_idx = np.arange(N, dtype=np.int64)
    order = np.lexsort((all_idx, -counts))

    pool_size = int(min(N, max(K_eff, 5 * K_eff)))
    top = order[:K_eff].astype(np.int64, copy=False)
    ann_elapsed = time.perf_counter() - t0
    remaining_time = deadline - time.perf_counter()
    can_run_hybrid = (
        pool_size > K_eff
        and processed_q == selected_queries.shape[0]
        and remaining_time >= max(2.0, 0.35 * ann_elapsed)
        and N <= 750_000
    )
    if can_run_hybrid:
        cand_idx = order[:pool_size].astype(np.int64, copy=False)
        cand_vectors = np.ascontiguousarray(base[cand_idx], dtype=np.float32)

        exact_index = faiss.IndexFlatL2(d)
        exact_index.add(cand_vectors)

        k_ref = int(min(k_eff, pool_size))
        exact_counts = np.zeros(pool_size, dtype=np.int32)

        for start in range(0, selected_queries.shape[0], batch_size):
            if time.perf_counter() >= deadline:
                break
            end = min(start + batch_size, selected_queries.shape[0])
            _, I_ref = exact_index.search(selected_queries[start:end], k_ref)
            valid_ref = I_ref[I_ref >= 0]
            if valid_ref.size:
                np.add.at(exact_counts, valid_ref.astype(np.int64, copy=False), 1)

        ann_counts_cand = counts[cand_idx].astype(np.int64, copy=False)
        order_cand = np.lexsort((cand_idx, -ann_counts_cand, -exact_counts))
        top = cand_idx[order_cand[:K_eff]].astype(np.int64, copy=False)

    if top.shape[0] < K_eff:
        chosen = np.zeros(N, dtype=bool)
        chosen[top] = True
        fill = all_idx[~chosen][: K_eff - top.shape[0]]
        top = np.concatenate([top, fill])

    return top[:K_eff]