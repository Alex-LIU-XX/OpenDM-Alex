#!/usr/bin/env bash
# OpenDM conda 环境安装脚本（按 README/Dockerfile 的方式：conda create -n opendm python=3.10 + pip install -e .）
# 注意：本容器出网必须绕过本地 http_proxy(127.0.0.1:64500)，否则连接会被 reset。
set -euo pipefail

CONDA=/root/miniconda3/bin/conda
ENV_PY=/root/miniconda3/envs/opendm/bin/python
REPO=/root/workspace/opendm

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export PIP_TRUSTED_HOST=mirrors.aliyun.com
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PATH=/root/miniconda3/bin:$PATH

step() { echo; echo "########## $* ##########"; date '+%F %T'; }

step "STEP 1/5 conda create -n opendm python=3.10"
if [ -x "$ENV_PY" ]; then
  echo "env opendm already exists, skip create"
else
  "$CONDA" create -n opendm python=3.10 -y
fi

step "STEP 2/5 upgrade pip"
"$ENV_PY" -m pip install --upgrade pip setuptools wheel

step "STEP 3/5 install torch 2.11.0 + torchvision 0.26.0 (cu128)"
"$ENV_PY" -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128

step "STEP 4/5 pip install -e . (repo deps)"
cd "$REPO"
"$ENV_PY" -m pip install -e .

step "STEP 5/5 verify"
"$ENV_PY" -c "import sys, torch, torchvision; print('python', sys.version.split()[0]); print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available()); print('torchvision', torchvision.__version__)"
"$ENV_PY" -m pip show OpenDM | head -8

echo
echo "ALL STEPS DONE"
