#!/usr/bin/env bash

# Usage: bash forest_fire.sh <graph_path> <seed_path> <output_path> <k> <num_sim> <hops>

if [ "$#" -ne 6 ]; then
    echo "Usage: bash forest_fire.sh <graph_file> <seed_file> <output_file> <k> <num_sim> <hops>"
    exit 1
fi

GRAPH_FILE="$1"
SEED_FILE="$2"
OUTPUT_FILE="$3"
K_BUDGET="$4"
NUM_SIM="$5"
HOPS="$6"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$(dirname "$OUTPUT_FILE")"

g++ -O3 -std=c++17 "${SCRIPT_DIR}/solver.cpp" -o "${SCRIPT_DIR}/solver"

"${SCRIPT_DIR}/solver" "$GRAPH_FILE" "$SEED_FILE" "$OUTPUT_FILE" "$K_BUDGET" "$NUM_SIM" "$HOPS"