#!/usr/bin/env bash
# 安装 flash-attn（可选 attention 层）。本环境踩到三个坑，脚本里都做了处理：
#
#   1) pip 直接从 aliyun 拉 sdist 会卡在 ~35 KB/s 的慢连接上（同一 URL 用 curl 重开
#      连接能跑 3.6 MB/s）—— 先用 curl 拉全，再 `pip install <sdist 路径>`。
#   2) setup.py 默认先去 GitHub releases 找预编译 wheel（setup.py:56），容器里
#      github.com:443 能建连但永不返回，进程会无限期挂住 —— 必须
#      FLASH_ATTENTION_FORCE_BUILD=TRUE 强制本地编译。
#   3) 容器 cgroup 内存上限只有 119 GiB（宿主机 1 TB），而 setup.py 默认按
#      sm80/90/100/120 四套架构生成 gencode，2026-09-21 那次 MAX_JOBS=16 直接把
#      上限打爆（多个 nvcc 被 OOM killer `Killed` → ninja exit 255）。
#      —— 必须用 FLASH_ATTN_CUDA_ARCHS 限成 80（本机只有 A100），并下调 MAX_JOBS。
#      ⚠️ 注意：架构开关是 FLASH_ATTN_CUDA_ARCHS，不是 TORCH_CUDA_ARCH_LIST！
#      setup.py:70 `os.getenv("FLASH_ATTN_CUDA_ARCHS", "80;90;100;120")` 只认前者，
#      设 TORCH_CUDA_ARCH_LIST 完全无效（上次就是这么白编了四套架构）。
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export PATH=/root/miniconda3/envs/opendm/bin:$PATH
export MAX_JOBS="${MAX_JOBS:-8}"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-80}"
PY=/root/miniconda3/envs/opendm/bin/python
SMOKE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # .smoke/ 目录
SDIST="$SMOKE/artifacts/flash_attn-2.8.3.tar.gz"

if [ ! -f "$SDIST" ]; then
  echo "缺少 sdist: $SDIST"
  echo "先按 .smoke/README.md 的方式用 curl 拉到该路径（pip 直连会卡在慢连接上）。"
  exit 1
fi

echo "[$(date '+%F %T')] MAX_JOBS=$MAX_JOBS FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=$FLASH_ATTN_CUDA_ARCHS SDIST=$SDIST"
"$PY" -u -m pip install "$SDIST" --no-build-isolation --no-deps -v
rc=$?
echo "[$(date '+%F %T')] pip rc=$rc"
if [ "$rc" -eq 0 ]; then
  "$PY" -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
fi
exit "$rc"
