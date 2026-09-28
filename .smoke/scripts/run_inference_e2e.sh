#!/usr/bin/env bash
# 用真实 Dexmal/DM05 权重跑端到端推理：起 HTTP 服务 -> 打 /v1/infer -> 收尾
#
# 关键点：DM05ModelConfig 默认 vision_attn_implementation=flash_attention_2，
# 本环境没装 flash-attn，所以显式降到 sdpa（llm 侧 flex_attention 由 torch 2.11 提供，可用）。
set -uo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

REPO=/root/workspace/opendm
PY=/root/miniconda3/envs/opendm/bin/python
SMOKE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # .smoke/ 目录
PORT="${PORT:-7891}"
LOG="$SMOKE/logs/inference_server.log"
# 权重本体在 CFS（本地 checkpoints/DM05 已删，见 docs/Alex/DM0.5_weights_on_CFS.md）
CKPT="${CKPT:-/mnt/cfs/opendm/checkpoints/DM05-sdpa-verify}"

mkdir -p "$SMOKE/logs" "$SMOKE/artifacts"

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/envs/opendm/bin:$PATH"

# 清掉上一轮残留
pkill -f 'dm05_exp[.]py' 2>/dev/null
sleep 1

echo "=== starting inference server (log: $LOG) ==="
: > "$LOG"
setsid nohup "$PY" opendm/exp/dm05_exp.py \
  --task inference \
  --model-config.model-name-or-path "$CKPT" \
  --model-config.chunk-size 50 \
  --model-config.vision-attn-implementation sdpa \
  --inference-config.output-action-dim 14 \
  --inference-config.image-prompts "Head" "Left wrist" "Right wrist" \
  --inference-config.port "$PORT" \
  >> "$LOG" 2>&1 < /dev/null &
SERVER_PID=$!
echo "launcher pid(setsid leader)=$SERVER_PID"

cleanup() {
  echo "=== stopping inference server ==="
  pkill -f 'dm05_exp[.]py' 2>/dev/null
  sleep 2
  pkill -9 -f 'dm05_exp[.]py' 2>/dev/null
}
trap cleanup EXIT

echo "=== waiting for port $PORT (model load can take a few minutes) ==="
READY=0
for i in $(seq 1 240); do
  if ! pgrep -f 'dm05_exp[.]py' > /dev/null; then
    echo "!!! server process died at t=${i}s; tail of log:"
    tail -60 "$LOG"
    exit 1
  fi
  if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:${PORT}/v1/infer" 2>/dev/null; then
    READY=1
    echo "=== port open after ${i}s ==="
    break
  fi
  if [ $((i % 20)) -eq 0 ]; then
    echo "  ... still loading (${i}s)"
  fi
  sleep 1
done

if [ "$READY" -ne 1 ]; then
  echo "!!! server never became ready; tail of log:"
  tail -80 "$LOG"
  exit 1
fi

sleep 2
echo
echo "=== resource usage ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv

echo
echo "=== POST /v1/infer (tests/curl_demo.sh) ==="
bash tests/curl_demo.sh "http://127.0.0.1:${PORT}/v1/infer" "DOS W1" | tee "$SMOKE/artifacts/infer_response.json"
echo
echo "=== second request (should hit CUDA Graph replay path) ==="
bash tests/curl_demo.sh "http://127.0.0.1:${PORT}/v1/infer" "Aloha"

echo
echo "=== server log tail ==="
tail -30 "$LOG"
