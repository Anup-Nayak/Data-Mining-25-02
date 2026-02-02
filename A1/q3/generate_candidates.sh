#!/bin/bash

DB_FEATURES=$1
QUERY_FEATURES=$2
OUT_FILE=$3

python3 generate_candidates.py "$DB_FEATURES" "$QUERY_FEATURES" "$OUT_FILE"
