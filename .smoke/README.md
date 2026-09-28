# .smoke —— OpenDM 环境搭建与冒烟测试工作区

本目录是**一次性验证工作区**，不是仓库的正式代码：装环境、下权重、跑冒烟、验端到端推理
过程中用到的脚本与留痕都集中在这里，和 `opendm/` 主代码隔开，方便随时整目录删除。

背景、踩坑过程与实测数据见 `docs/Alex/opendm_env_setup_and_smoke_test.md`。

## 目录约定

| 目录 | 内容 | 是否入库 |
| --- | --- | --- |
| `scripts/` | 可复用脚本（装环境、下权重、冒烟测试、端到端推理） | ✅ 入库 |
| `logs/` | 上述脚本跑出来的 `.log` 运行日志 | ❌ 本地留痕，已 gitignore |
| `artifacts/` | 下载缓存与接口响应等产物（如 flash-attn sdist、`infer_response.json`） | ❌ 本地留痕，已 gitignore |

脚本内部统一用 `SMOKE="$(dirname "$0")/.."` 定位本目录，因此 `logs/`、`artifacts/`
**移动或清空都不影响脚本运行**（写入前会自行 `mkdir -p`）。

## scripts/ 一览

按推荐执行顺序：

| 脚本 | 作用 |
| --- | --- |
| `install_env.sh` | 建 conda 环境 `opendm`（py3.10）+ torch 2.11.0/cu128 + `pip install -e .`；内置绕过容器代理、切 aliyun 源。约 40 分钟 |
| `download_ckpt.sh` | 用 `hf download` 从 hf-mirror 拉 `Dexmal/DM05`（官方姿势，本网络下约 60+ 分钟） |
| `parallel_download.py` | 同一份权重的并发分片下载器：44×256 MB、8 线程 Range、逐片长度校验、拼接后校验 SHA256。实测 2.9 分钟 / 63.6 MB/s |
| `install_flash_attn.sh` | 装可选依赖 flash-attn（sdist 本地编译）；内置三个坑的处理：慢连接改用 curl 预拉、`FLASH_ATTENTION_FORCE_BUILD=TRUE` 避开挂死的 GitHub release 探测、限定 `TORCH_CUDA_ARCH_LIST=8.0` 防 OOM |
| `run_smoke.sh` | 冒烟总入口：`pytest tests/` → CPU 微型模型前向/推理 → 训练入口 `--help`，最后打印汇总 |
| `smoke_test.py` | CPU 冒烟本体：随机初始化微型 DM05（fp32_mixed），验证构建、forward+backward、`inference_action`、参数量统计 |
| `smoke_test_gpu.py` | GPU 冒烟本体（A100）：bf16 autocast 训练路径 + `sdpa`(CUDA Graph) / `eager` 两种后缀注意力推理分支 |
| `run_inference_e2e.sh` | 真实权重端到端：起 HTTP 推理服务 → 等端口 → 打两次 `/v1/infer`（含 CUDA Graph replay）→ 收尾；响应存 `artifacts/infer_response.json` |

## 常用命令

```bash
# 关键前置：本容器出网必须绕过预置代理，否则连接被 reset
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

cd /root/workspace/opendm

# 冒烟（CPU + 训练入口）
bash .smoke/scripts/run_smoke.sh

# GPU 冒烟
/root/miniconda3/envs/opendm/bin/python .smoke/scripts/smoke_test_gpu.py

# 真实权重端到端推理（权重走 CFS，可用 CKPT=... 覆盖；PORT=... 换端口）
bash .smoke/scripts/run_inference_e2e.sh
```

## 当前权重位置

本地 `checkpoints/DM05` 已删除（2026-09-21，释放 11 G），唯一副本在
**`/mnt/cfs/opendm/checkpoints/`**；`run_inference_e2e.sh` 默认用的就是
`/mnt/cfs/opendm/checkpoints/DM05-sdpa-verify`。细节见 `docs/Alex/DM0.5_weights_on_CFS.md`。
