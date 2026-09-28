#!/usr/bin/env bash
# OpenDM 冒烟测试：单元测试 + 微型模型前向/推理 + 训练入口 CLI
set -uo pipefail

ENV_PY=/root/miniconda3/envs/opendm/bin/python
REPO=/root/workspace/opendm
SMOKE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # .smoke/ 目录

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true
export PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$REPO"

step() { echo; echo "########## $* ##########"; date '+%F %T'; }

step "SMOKE 1/3 单元测试 pytest tests/ -v"
"$ENV_PY" -m pytest tests/ -v
RC1=$?

step "SMOKE 2/3 微型 DM05 前向 + 动作推理"
"$ENV_PY" "$SMOKE/scripts/smoke_test.py"
RC2=$?

step "SMOKE 3/3 训练入口 CLI (tyro --help)"
timeout 300 "$ENV_PY" -m opendm.exp.dm05_exp --help > /tmp/dm05_exp_help.txt 2>&1
RC3=$?
if [ $RC3 -eq 0 ]; then
  echo "CLI --help OK, 前 25 行:"
  head -25 /tmp/dm05_exp_help.txt
else
  echo "CLI --help 失败 (rc=$RC3), 输出:"
  tail -30 /tmp/dm05_exp_help.txt
fi

echo
echo "==== SMOKE SUMMARY ===="
echo "pytest           rc=$RC1"
echo "model smoke      rc=$RC2"
echo "cli --help       rc=$RC3"
[ $RC1 -eq 0 ] && [ $RC2 -eq 0 ] && [ $RC3 -eq 0 ] && echo "OVERALL: OK" || echo "OVERALL: FAILED"
