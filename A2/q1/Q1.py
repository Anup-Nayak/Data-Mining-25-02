#!/usr/bin/env python3
"""
Usage:
    python3 Q1.py <dataset_num>          # mode-1: when a number is provided -> run for BOTH datasets 1 & 2 and produce one plot.png with two subplots
    python3 Q1.py <path_to_dataset>.npy  # mode-2: load a single .npy file and produce one plot.png with one subplot
"""
import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import urllib.request
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


def load_from_api(dataset_num: int) -> np.ndarray:
    student_id = "cs5221443"
    url = (
        f"http://hulk.cse.iitd.ac.in:3000/dataset"
        f"?student_id={student_id}&dataset_num={dataset_num}"
    )
    with urllib.request.urlopen(url) as response:
        raw_data = response.read().decode("utf-8")
    data = json.loads(raw_data)
    return np.array(data["X"])


def load_from_npy(path: str) -> np.ndarray:
    return np.load(path)


def compute_inertia(X: np.ndarray, k: int, n_init: int = 10, random_state: int = 42) -> float:
    km = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=n_init,
        max_iter=500,
        tol=1e-5,
        random_state=random_state,
    )
    km.fit(X)
    return float(km.inertia_)


def find_elbow(ks: list, inertias: list) -> int:
    ks_arr = np.array(ks, dtype=float)
    ins_arr = np.array(inertias, dtype=float)

    if len(ks_arr) <= 1:
        return int(ks_arr[0]) if len(ks_arr) == 1 else 1

    x = (ks_arr - ks_arr.min()) / (ks_arr.max() - ks_arr.min())
    y = (ins_arr - ins_arr.min()) / (ins_arr.max() - ins_arr.min())

    dx = x[-1] - x[0]
    dy = y[-1] - y[0]
    line_len = np.hypot(dx, dy)

    if line_len == 0:
        return int(ks[0])

    distances = np.abs(dy * x - dx * y + x[-1] * y[0] - y[-1] * x[0]) / line_len
    elbow_idx = int(np.argmax(distances))
    return int(ks[elbow_idx])


def make_plots_multi(ks: list, inertias_list: list, optimal_ks: list,
                     dataset_labels: list, out_path: str = "plot.png",
                     mode_label: str = "mode 1") -> None:
    n = len(inertias_list)
    fig, axes = plt.subplots(1, n, figsize=(6 * max(1, n), 5), squeeze=False)
    axes = axes[0]

    for i in range(n):
        ax = axes[i]
        ax.plot(ks, inertias_list[i], marker="o", linewidth=2, markersize=5)
        opt_k = optimal_ks[i]
        try:
            opt_inertia = inertias_list[i][ks.index(opt_k)]
        except ValueError:
            opt_inertia = None
        ax.axvline(x=opt_k, color="red", linestyle="--", linewidth=1.2)
        if opt_inertia is not None:
            ax.scatter([opt_k], [opt_inertia], color="red", s=80, zorder=5)
        ax.set_title(f"{dataset_labels[i]}", fontsize=12)
        ax.set_xlabel("Number of clusters (k)", fontsize=11)
        ax.set_ylabel("k-means Objective (Inertia)", fontsize=11)
        ax.set_xticks(ks)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(f"{mode_label}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_single_plot(ks: list, inertias: list, optimal_k: int, out_path: str = "plot.png", mode_label: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, inertias, marker="o", linewidth=2, markersize=6)
    try:
        opt_inertia = inertias[ks.index(optimal_k)]
    except ValueError:
        opt_inertia = None
    ax.axvline(x=optimal_k, color="red", linestyle="--", linewidth=1.2)
    if opt_inertia is not None:
        ax.scatter([optimal_k], [opt_inertia], color="red", s=100, zorder=5)
    ax.set_xlabel("Number of clusters (k)", fontsize=13)
    ax.set_ylabel("k-means Objective (Inertia)", fontsize=13)
    ax.set_title(mode_label if mode_label else "k-means Objective vs. Number of Clusters", fontsize=14, fontweight="bold")
    ax.set_xticks(ks)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_on_dataset(X: np.ndarray, K_RANGE: list, n_init: int):
    inertias = []
    for k in K_RANGE:
        inertia = compute_inertia(X, k, n_init=n_init)
        inertias.append(inertia)
    optimal_k = find_elbow(K_RANGE, inertias)
    return inertias, optimal_k


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 Q1.py <dataset_num>  OR  python3 Q1.py <path>.npy", file=sys.stderr)
        sys.exit(1)

    arg = sys.argv[1]

    K_RANGE = list(range(1, 16))

    inertias_results = []
    optimal_ks = []
    labels = []

    if arg.endswith(".npy"):
        try:
            X = load_from_npy(arg)
        except Exception as e:
            print(f"Error loading .npy file: {e}", file=sys.stderr)
            sys.exit(1)

        assert isinstance(X, np.ndarray), "Loaded data must be a numpy array."
        assert X.ndim == 2, f"Expected 2-D array, got shape {X.shape}."

        n_samples, _ = X.shape
        n_init = max(5, min(15, 50_000 // max(n_samples, 1)))

        inertias, opt_k = run_on_dataset(X, K_RANGE, n_init=n_init)
        optimal_ks.append(opt_k)
        inertias_results.append(inertias)
        labels.append("Dataset (from .npy)")
        make_single_plot(K_RANGE, inertias, opt_k, out_path=os.path.join(os.getcwd(), "plot.png"),
                         mode_label="mode: file (single dataset)")
        print(opt_k)

    else:
        try:
            _ = int(arg)
            dataset_nums = [1, 2]
        except ValueError:
            print(f"Error: argument '{arg}' is not a valid dataset number or .npy path.", file=sys.stderr)
            sys.exit(1)

        for dnum in dataset_nums:
            try:
                X = load_from_api(dnum)
            except Exception as e:
                print(f"Error fetching dataset {dnum} from API: {e}", file=sys.stderr)
                sys.exit(1)

            assert isinstance(X, np.ndarray), "Loaded data must be a numpy array."
            assert X.ndim == 2, f"Expected 2-D array, got shape {X.shape}."

            n_samples, _ = X.shape
            n_init = max(5, min(15, 50_000 // max(n_samples, 1)))

            inertias, opt_k = run_on_dataset(X, K_RANGE, n_init=n_init)
            inertias_results.append(inertias)
            optimal_ks.append(opt_k)
            labels.append(f"Dataset {dnum}")

        out_path = os.path.join(os.getcwd(), "plot.png")
        make_plots_multi(K_RANGE, inertias_results, optimal_ks, dataset_labels=labels,
                         out_path=out_path, mode_label="mode 1: API datasets (1 & 2)")

        for idx, dnum in enumerate(dataset_nums):
            print(f"Dataset {dnum} optimal k: {optimal_ks[idx]}")


if __name__ == "__main__":
    main()