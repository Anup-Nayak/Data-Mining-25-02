#!/bin/bash

DATASET_PATH=$1
OUT_FEATURE_PATH=$2

TMP_GSPAN_DATA=./data_gspan.dat
GSPAN_FP=./data_gspan.dat.fp

python3 preprocess_gspan.py "$DATASET_PATH" "$TMP_GSPAN_DATA"

./gSpan -f "$TMP_GSPAN_DATA" -s 0.15 -o -i

python3 identify.py "$GSPAN_FP" "$OUT_FEATURE_PATH"

rm -rf "$TMP_GSPAN_DATA" "$GSPAN_FP"
