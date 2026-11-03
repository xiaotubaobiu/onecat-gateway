#!/usr/bin/env bash
# usage: run-mode.sh <target|dflash|dflash8k|tuned|dflash-tuned>
set -euo pipefail
MODE="${1:?usage: run-mode.sh <target|dflash|dflash8k|tuned|dflash-tuned>}"
case "$MODE" in target|dflash|dflash8k|tuned|dflash-tuned) ;; *) echo "unknown mode $MODE"; exit 1;; esac
echo "$MODE" > /home/z/onecat/config/mode
docker compose -f /home/z/onecat/compose/docker-compose.yml up -d --force-recreate 2>&1 | tail -1
echo "mode=$MODE; log: docker logs -f onecat-vllm"
