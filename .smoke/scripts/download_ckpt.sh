#!/usr/bin/env bash
# 下载 Dexmal/DM05 预训练权重（走 hf-mirror，必须绕过容器预置代理）
set -euo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

export HF_ENDPOINT=https://hf-mirror.com
# hf-mirror 不支持 xet 协议，强制走普通 HTTP 分片下载
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME=/root/workspace/opendm/checkpoints/.hf_home

PY=/root/miniconda3/envs/opendm/bin/python
HF=/root/miniconda3/envs/opendm/bin/hf

DEST=/root/workspace/opendm/checkpoints/DM05
mkdir -p "$DEST"

echo "[$(date '+%F %T')] start download Dexmal/DM05 -> $DEST"
"$HF" download Dexmal/DM05 --local-dir "$DEST" --max-workers 8
echo "[$(date '+%F %T')] download finished rc=$?"
du -sh "$DEST"
