#!/bin/bash

GPU_ID=0
MODEL_PATH="./models/Dream-v0-Instruct-7B"
SAVE_DIR="./results"

GEN_LENGTH=256
STEPS=256

DATASET="gsm8k"

ALG="klass"
UNMASK_STRATEGY="all"

CONF_THRESHOLD=0.9
KL_THRESHOLD=0.001
HISTORY_LENGTH=2

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./src/dream_evaluation.py \
  --model_path "$MODEL_PATH" \
  --save_dir "$SAVE_DIR" \
  --gen_length $GEN_LENGTH \
  --steps $STEPS \
  --conf_threshold $CONF_THRESHOLD \
  --kl_threshold $KL_THRESHOLD \
  --history_length $HISTORY_LENGTH \
  --dataset $DATASET \
  --unmask_strategy $UNMASK_STRATEGY \
  --alg $ALG \
  --save_steps