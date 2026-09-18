#!/usr/bin/env bash
set -euo pipefail
: "${VLLM_API_KEY:?VLLM_API_KEY is not set}"
MAX_SEQS="${ONCAT_MAX_SEQS:-4}"
BUDGET="${ONCAT_BUDGET:-4096}"
MLEN="${ONCAT_MAX_MODEL_LEN:-32768}"
UTIL="${ONCAT_UTIL:-0.80}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
echo "tuned params: max_num_seqs=$MAX_SEQS budget=$BUDGET util=$UTIL"
exec /home/z/onecat/venv/bin/vllm serve \
  /Dataset/onecat/models/Qwen3.8-27B-QUASAR-NVFP4 \
  --served-model-name qwen3.8-27b \
  --trust-remote-code \
  --dtype half \
  --tensor-parallel-size 4 \
  --attention-backend FLASH_ATTN_V100 \
  --kv-cache-dtype fp8_e5m2 \
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
  --api-key "${VLLM_API_KEY}" \
  --host 127.0.0.1 \
  --port 8000
