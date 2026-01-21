#!/bin/bash

SRC_MODEL=../models/mt5-translit
OUT_DIR=../models/ct2-mt5

mkdir -p $OUT_DIR

ct2-transformers-converter \
  --model $SRC_MODEL \
  --output_dir $OUT_DIR \
  --quantization int8 \
  --force
