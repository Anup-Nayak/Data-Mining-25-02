if [ "$#" -ne 5 ]; then
    echo "Usage:"
    echo "  bash q2.sh <path_gspan_executable> <path_fsg_executable> \\"
    echo "            <path_gaston_executable> <path_dataset> <path_out>"
    exit 1
fi


GSPAN_BIN="$1"
FSG_BIN="$2"
GASTON_BIN="$3"
DATASET="$4"
OUTROOT="$5"


MIN_SUPS=(5 10 25 50 95)

TIME_LIMIT="1h"

WORKDIR="${OUTROOT}/work"
PLOTDIR="${OUTROOT}"
CSV="${OUTROOT}/runtime_results.csv"

mkdir -p "$WORKDIR"

echo "algorithm,minsup,runtime_seconds" > "$CSV"

echo "Converting dataset formats..."

python3 yeast_fsg.py     "$DATASET" "$WORKDIR/yeast_fsg.txt"
python3 yeast_gaston.py   "$DATASET" "$WORKDIR/yeast_gspan.txt"
python3 yeast_gaston.py  "$DATASET" "$WORKDIR/yeast_gaston.txt"


NUM_GRAPHS=$(grep -c "^t" "$WORKDIR/yeast_gaston.txt")
echo "Number of graphs: $NUM_GRAPHS"
echo

for SUP in "${MIN_SUPS[@]}"; do
    echo "======================================"
    echo "Running experiments for minSup = ${SUP}%"
    echo "======================================"

    GSPAN_SUP=$(echo "$SUP / 100" | bc -l)
    GASTON_SUP=$(echo "($SUP * $NUM_GRAPHS + 99) / 100" | bc)  # ceil

    echo "Running FSG..."
    START=$(date +%s.%N)

    timeout "$TIME_LIMIT"  "$FSG_BIN" -s "$SUP"  "$WORKDIR/yeast_fsg.txt" > "$OUTROOT/fsg$SUP.fp" 

    STATUS=$?
    END=$(date +%s.%N)

    if [ "$STATUS" -eq 124 ]; then
        echo "FSG timed out"
        echo "FSG,$SUP,TIMEOUT" >> "$CSV"
    else
        TIME=$(echo "$END - $START" | bc)
        echo "FSG,$SUP,$TIME" >> "$CSV"
    fi

    echo "Running gSpan..."
    START=$(date +%s.%N)

    timeout "$TIME_LIMIT" "$GSPAN_BIN"  -f "$WORKDIR/yeast_gspan.txt"  -s "$GSPAN_SUP" -o -i > "$OUTROOT/gspan$SUP.txt"
    mv "$WORKDIR/yeast_gspan.txt.fp" "$OUTROOT/gspan$SUP.fp"

    STATUS=$?
    END=$(date +%s.%N)

    if [ "$STATUS" -eq 124 ]; then
        echo "gSpan timed out"
        echo "gSpan,$SUP,TIMEOUT" >> "$CSV"
    else
        TIME=$(echo "$END - $START" | bc)
        echo "gSpan,$SUP,$TIME" >> "$CSV"
    fi

    echo "Running Gaston..."
    START=$(date +%s.%N)

    timeout "$TIME_LIMIT" "$GASTON_BIN" "$GASTON_SUP"  "$WORKDIR/yeast_gaston.txt" "$OUTROOT/gaston$SUP.out"

    STATUS=$?
    END=$(date +%s.%N)

    if [ "$STATUS" -eq 124 ]; then
        echo "Gaston timed out"
        echo "Gaston,$SUP,TIMEOUT" >> "$CSV"
    else
        TIME=$(echo "$END - $START" | bc)
        echo "Gaston,$SUP,$TIME" >> "$CSV"
    fi

    echo
done


echo "Generating plot..."

python3 - <<EOF
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("$CSV")

df = df[df.runtime_seconds != "TIMEOUT"]
df["runtime_seconds"] = df["runtime_seconds"].astype(float)

for algo in df.algorithm.unique():
    sub = df[df.algorithm == algo]
    plt.plot(sub.minsup, sub.runtime_seconds, marker='o', label=algo)

plt.xlabel("Minimum Support (%)")
plt.ylabel("Runtime (seconds)")
plt.title("FSG vs gSpan vs Gaston on Yeast Dataset")
plt.legend()
plt.grid(True)
plt.savefig("$OUTROOT/runtime_comparison.png", dpi=300)
EOF

echo "======================================"
echo "All experiments completed!"
echo "Plot: $OUTROOT/runtime_comparison.png"
echo "CSV : $CSV"
echo "======================================"
