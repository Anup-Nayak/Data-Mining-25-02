#!/usr/bin/env python3

import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
    print("Usage: python plot_q1_1.py <runtime_csv> <output_plot_path>")
    sys.exit(1)

csv_path = sys.argv[1]
out_plot = sys.argv[2]

df = pd.read_csv(csv_path)
df = df.sort_values("support")

plt.figure(figsize=(8,5))
plt.plot(df["support"], df["apriori"], marker='o', label="Apriori")
plt.plot(df["support"], df["fpgrowth"], marker='o', label="FP-Growth")

plt.xlabel("Support Threshold (%)")
plt.ylabel("Runtime (seconds)")
plt.title("Apriori vs FP-Growth Runtime Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_plot)
