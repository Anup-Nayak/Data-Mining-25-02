#!/bin/bash

GRAPH_PATH=$1
FEATURE_PATH=$2
OUT_PATH=$3

python3 convert.py "$GRAPH_PATH" "$FEATURE_PATH" "$OUT_PATH"
