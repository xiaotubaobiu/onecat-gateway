#!/usr/bin/env bash
# Production DFlash2 deployment; physical GPUs 4-7 are exposed as container 0-3.
set -euo pipefail
: "${VLLM_API_KEY:?VLLM_API_KEY is not set}"
MODEL="${ONCAT_MODEL_PATH:-/Dataset/onecat/models/Qwen3.8-27B-Uncensored-NVFP4-1Cat}"
DRAFT="${ONCAT_DRAFT_PATH:-/Dataset/onecat/models/Qwen3.8-27B-DFlash2}"
MAX_SEQS="${ONCAT_MAX_SEQS:-6}"
BUDGET="${ONCAT_BUDGET:-8192}"
MLEN="${ONCAT_MAX_MODEL_LEN:-262144}"
UTIL="${ONCAT_UTIL:-0.80}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
SPECULATIVE_CONFIG=$(/home/z/onecat/venv/bin/python -c \
  'import json,sys; print(json.dumps({"method":"dflash","model":sys.argv[1],"kv_cache_dtype":"auto"}))' "$DRAFT")
echo "DFlash2 tuned: model=$MODEL max_num_seqs=$MAX_SEQS budget=$BUDGET util=$UTIL context=$MLEN"
exec /home/z/onecat/venv/bin/vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --trust-remote-code \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --attention-backend FLASH_ATTN_V100 \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len "$MLEN" \
  --gpu-memory-utilization "$UTIL" \
  --max-num-batched-tokens "$BUDGET" \
  --max-num-seqs "$MAX_SEQS" \
  --limit-mm-per-prompt '{"image": 8}' \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --mamba-cache-mode align \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --speculative-config "$SPECULATIVE_CONFIG" \
  --api-key "$VLLM_API_KEY" \
  --host 127.0.0.1 \
  --port 8000
