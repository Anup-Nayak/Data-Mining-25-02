#!/usr/bin/env bash

set -e

if [ "$#" -ne 4 ]; then
    echo "Usage: bash q1_1.sh <path_apriori_executable> <path_fp_executable> <path_dataset> <path_out>"
    exit 1
fi

APRIORI_BIN="$1"
FP_BIN="$2"
DATASET="$3"
OUT_DIR="$4"

SUPPORTS=(5 10 25 50 90)

mkdir -p "$OUT_DIR"

RUNTIME_FILE="$OUT_DIR/runtime.csv"
echo "support,apriori,fpgrowth" > "$RUNTIME_FILE"

for s in "${SUPPORTS[@]}"; do
    echo "Running support ${s}%..."

    AP_OUT="$OUT_DIR/ap${s}"
    FP_OUT="$OUT_DIR/fp${s}"

    # -------------------------
    # APRIORI
    # -------------------------
    start=$(date +%s.%N)

    set +e
    timeout 3600 "$APRIORI_BIN" -s${s} "$DATASET" "$AP_OUT" > /dev/null 2>&1
    exit_code=$?
    set -e

    end=$(date +%s.%N)

    if [ ! -f "$AP_OUT" ]; then
        : > "$AP_OUT"
    fi

    if [ $exit_code -eq 124 ]; then
        ap_time=3600
    else
        ap_time=$(awk "BEGIN {printf \"%.6f\", $end - $start}")
    fi

    # -------------------------
    # FP-GROWTH
    # -------------------------
    start=$(date +%s.%N)

    set +e
    timeout 3600 "$FP_BIN" -s${s} "$DATASET" "$FP_OUT" > /dev/null 2>&1
    exit_code=$?
    set -e

    end=$(date +%s.%N)
    
    if [ ! -f "$FP_OUT" ]; then
    : > "$FP_OUT"
    fi

    if [ $exit_code -eq 124 ]; then
        fp_time=3600
    else
        fp_time=$(awk "BEGIN {printf \"%.6f\", $end - $start}")
    fi

    echo "${s},${ap_time},${fp_time}" >> "$RUNTIME_FILE"
done

python3 "$(dirname "$0")/plot_q1_1.py" "$RUNTIME_FILE" "$OUT_DIR/plot.png"
