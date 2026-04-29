#!/usr/bin/env bash
set -euo pipefail
RAW_DIR=/root/lanyun-fs/tokenizedsae_l22_sidecar/raw_token_ids/layer_22
LOG_DIR=/root/lanyun-fs/tokenizedsae_l22_sidecar/logs
TARGET_COUNT=50000
while true; do
  count=$(find "$RAW_DIR" -maxdepth 1 -name 'token_ids_idx*.pt' | wc -l)
  echo "[$(date '+%F %T')] raw_token_count=$count" >> "$LOG_DIR/wait_and_build_paired.log"
  if [ "$count" -ge "$TARGET_COUNT" ]; then
    break
  fi
  if ! pgrep -f 'build_token_ids_raw_sidecar.py' >/dev/null; then
    echo "[$(date '+%F %T')] raw token generation stopped early at count=$count" >> "$LOG_DIR/wait_and_build_paired.log"
    exit 1
  fi
  sleep 60
done
echo "[$(date '+%F %T')] starting paired batch build" >> "$LOG_DIR/wait_and_build_paired.log"
cd /root/saint
eval "$(poetry env activate)"
python tokenizedsae_sidecar_tools/build_paired_tokenized_batches.py \
  --raw_activation_dir /root/saint/activation_outputs/layer_22 \
  --raw_token_ids_dir /root/lanyun-fs/tokenizedsae_l22_sidecar/raw_token_ids/layer_22 \
  --output_dir /root/lanyun-fs/tokenizedsae_l22_sidecar/paired_batches \
  --mean_filepath /root/lanyun-fs/tokenizedsae_l22_sidecar/activation_outputs_mean_tokenized.pt \
  --num_processes 20 \
  --batch_size 2048 \
  > /root/lanyun-fs/tokenizedsae_l22_sidecar/logs/build_paired_tokenized_batches.log 2>&1
echo "[$(date '+%F %T')] paired batch build finished" >> "$LOG_DIR/wait_and_build_paired.log"
