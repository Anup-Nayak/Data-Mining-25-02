#!/usr/bin/env bash

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: bash q1_2.sh <universal_itemset_size> <num_transactions>"
    exit 1
fi

UNIV_SIZE="$1"
NUM_TXNS="$2"

python3 generate_dataset.py "$UNIV_SIZE" "$NUM_TXNS"

if [ ! -f generated_transactions.dat ]; then
    echo "Error: Dataset generation failed."
    exit 1
fi
