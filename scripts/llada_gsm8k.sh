#!/bin/bash

GPU_ID=0
MODEL_PATH="./models/LLaDA-8B-Instruct"
SAVE_DIR="./results"

GEN_LENGTH=256
BLOCK_LENGTH=64
STEPS=256

DATASET="gsm8k"

ALG="klass"
UNMASK_STRATEGY="all"

CONF_THRESHOLD=0.6
KL_THRESHOLD=0.015
HISTORY_LENGTH=2

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./src/llada_evaluation.py \
  --model_path "$MODEL_PATH" \
  --save_dir "$SAVE_DIR" \
  --gen_length $GEN_LENGTH \
  --block_length $BLOCK_LENGTH \
  --steps $STEPS \
  --conf_threshold $CONF_THRESHOLD \
  --kl_threshold $KL_THRESHOLD \
  --history_length $HISTORY_LENGTH \
  --dataset $DATASET \
  --alg $ALG \
  --unmask_strategy $UNMASK_STRATEGY \
  --save_steps
