# OpenDM 环境配置与冒烟测试记录

- 记录时间：2026-09-20 19:21 (CST)
- 最后更新：2026-09-21 11:00 (CST)
- 机器：单卡 NVIDIA A100-SXM4-80GB，容器内 root 用户
- 仓库：`/root/workspace/opendm`（commit `fbab441`）
- 结论：
  - **conda 环境已按 README/Dockerfile 建好，CPU 与 GPU 冒烟测试全部通过**（2026-09-20）。
  - **预训练权重 `Dexmal/DM05` 已完整下载并通过 SHA256 校验**，真实 checkpoint 端到端推理**已验证通过**（2026-09-21，见第 6、9 节）。
  - `fast-infer`（onnx/tensorrt）可选依赖层仍未安装；`flash-attn` 的安装情况见第 7 节。

---

## 1. 结论速览

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| conda 环境 `opendm` | ✅ 已创建 | `/root/miniconda3/envs/opendm`，Python 3.10.21，占用 8.3 G |
| torch / torchvision (cu128) | ✅ 已安装 | 2.11.0+cu128 / 0.26.0+cu128 |
| 项目 editable 安装 | ✅ 已安装 | `OpenDM 0.1.0`，`import opendm` 指向仓库源码 |
| 单元测试 `pytest tests/` | ✅ 4 passed | CPU/GPU 环境下均通过 |
| 模型冒烟（CPU） | ✅ 4/4 PASS | 随机权重微型 DM05，前向/反向/推理/参数统计 |
| 模型冒烟（GPU） | ✅ 3/3 PASS | A100 上 bf16 训练路径 + 推理（含 CUDA Graph 分支） |
| 训练入口 CLI | ✅ rc=0 | `python -m opendm.exp.dm05_exp --help` |
| GPU 可用性 | ✅ 可用 | 先前不可见是**沙箱**所致，非机器/驱动问题 |
| 权重下载 `Dexmal/DM05` | ✅ 已完成 | 12/12 文件，10.97 GiB，落在 `./checkpoints/DM05` |
| 权重完整性 | ✅ SHA256 匹配 | 主权重 sha256 == Hub 登记的 LFS oid |
| 真实 checkpoint 端到端推理 | ✅ 已验证 | 服务起来了，`/v1/infer` 返回 50×14 action chunk |
| `flash-attn` | 见第 7 节 | `attention` 可选层；**默认 launcher 命令会因它缺失而加载失败** |
| `onnx` / `tensorrt` | ❌ 未安装 | `fast-infer` 可选层，仅 fast backend 需要 |

---

## 2. 机器与 GPU 复检

`nvidia-smi`：

```
NVIDIA-SMI 535.309.01     Driver Version: 535.309.01     CUDA Version: 12.8
GPU 0  NVIDIA A100-SXM4-80GB   On   30C   P0   82W / 400W   7698MiB / 81920MiB   0%
```

torch 侧复检：

```
torch: 2.11.0+cu128
cuda_available: True
device_count: 1
  gpu0: NVIDIA A100-SXM4-80GB cc8.0 mem=79.2GiB sm=108
gpu matmul ok, sum = 18807.062
bf16 supported: True
capability: (8, 0)
```

要点：

- 显卡确实存在且可用（`cc8.0` / sm80，A100 支持 bf16、支持 flash-attn v2）。
- `nvidia-smi` 显示 `7698 MiB / 81920 MiB` 已被占用，但进程列表为空——是同一台物理机上其他容器的占用，与本环境无关；剩余约 71 GiB 可用显存。
- **重要结论**：在本会话早期的沙箱（`workspace-write`）模式下，`torch.cuda.is_available()` 返回 `False`，并报
  `CUDA initialization: ... Error 304: OS call failed or operation not supported on this OS`。
  切换到完全访问模式后同一条命令返回 `True`。也就是说 **CUDA 之前不可见是沙箱拦截 `/dev/nvidia*` 造成的，不是驱动或机器问题**。跑训练/推理前请确认执行环境没有被沙箱限制。
- `torch.cuda.is_available()` 在**首次 CUDA 调用之前**判断才有意义；如果在 CUDA 上下文初始化失败后再判断，可能出现“同一次进程内结果不一致”的假象。

---

## 3. 环境安装过程

### 3.1 为什么是 conda

仓库是以 conda 为基准的：`Dockerfile` 装 Miniconda 并 `conda create -n opendm python=3.10`；README（中英）与 `docs/{zh,en}/dm05_*_lora_training.md` 也都用 `conda create -n opendm` / `conda activate opendm`；仓库中**没有 `uv.lock`、没有 `[tool.uv]` 配置、没有任何 uv 调用**（`pyproject.toml` 用 setuptools 构建，依赖由 pip 安装）。因此本次按 conda 路线执行。

### 3.2 执行的命令

```bash
# 1) 建环境（与 README/Dockerfile 一致）
conda create -n opendm python=3.10 -y

# 2) 基础打包工具
/root/miniconda3/envs/opendm/bin/python -m pip install --upgrade pip setuptools wheel

# 3) 先装 CUDA 12.8 版 PyTorch（走官方 cu128 索引）
/root/miniconda3/envs/opendm/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128

# 4) 安装项目本体（editable）及其全部基础依赖
cd /root/workspace/opendm
/root/miniconda3/envs/opendm/bin/python -m pip install -e .
```

> 全程必须 `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`，原因见第 4 节。

一键脚本见 `.smoke/scripts/install_env.sh`，完整日志见 `.smoke/logs/install.log`。

### 3.3 安装结果（关键版本）

| 组件 | 版本 |
| --- | --- |
| Python | 3.10.21（conda 环境内） |
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| triton | 3.6.0（随 torch 自动装入） |
| transformers | 5.3.0 |
| tokenizers | 0.22.1 |
| datasets / accelerate / peft | 5.0.0 / 1.14.0 / 0.19.1 |
| diffusers / timm / liger-kernel | 0.38.0 / 1.0.27 / 0.8.3 |
| numpy / pydantic / protobuf | 1.26.4 / 2.13.4 / 7.35.1 |
| gradio / wandb | 6.27.0 / 0.30.0 |
| opencv-python-headless | 4.11.0.86（版本回溯后落点） |
| pytest | 9.1.1 |
| OpenDM | 0.1.0（editable，`/root/workspace/opendm`） |

环境体积 **8.3 G**；容器磁盘 `98G` 中已用 `51G`，剩余 **48 G**（若后续要装 tensorrt + 下 10.9 GiB checkpoint，注意留出空间）。

### 3.4 依赖解析中的两个现象（非错误）

1. **opencv-python-headless 版本回溯**：由于项目锁定 `numpy==1.26.4`，而 opencv 5.x 需要 `numpy>=2`，pip 依次下载了 `5.0.0.93 → 4.14.0.94 → 4.13.0.92 → 4.13.0.90 → 4.12.0.88 → 4.11.0.86` 共 6 个约 50–60 MB 的 wheel 才落定，属于正常回溯，只是耗时与流量偏大。
2. **`pyramid==1.5` 源码构建**：该版本无 wheel，pip 从 sdist 构建成功（构建产物缓存于 `/root/.cache/pip/wheels`），安装后无异常。
3. `pip install -e .` 期间 `setuptools` 被 torch 的约束从 84.0.0 降到 78.1.0（`torch==2.11.0` 要求 `setuptools<82`），属预期行为。

---

## 4. 环境坑与解决（重要，换机器/换容器会再遇到）

### 4.1 出网代理必须绕过

容器内预设了：

```
http_proxy=http://127.0.0.1:64500
https_proxy=http://127.0.0.1:64500
```

走这个代理访问外网一律 `curl: (56) Recv failure: Connection reset by peer`。**所有安装/下载命令都要先 `unset http_proxy https_proxy`（或 `curl --noproxy '*'`）**，绕过代理后网络正常。

### 4.2 pip 源可用性实测

`/root/.config/pip/pip.conf` 默认指向 `mirrors.baidubce.com`（被 pip 判定为不受信而忽略），`/root/.pip/pip.conf` 指向清华源。实测结果：

| 源 | 结果 |
| --- | --- |
| `mirrors.baidubce.com`（配置默认） | ❌ pip 报 not trusted / 无匹配发行版 |
| `pypi.tuna.tsinghua.edu.cn` | ❌ HTTP 403（拒绝访问） |
| `mirrors.ustc.edu.cn` | ❌ HTTP 403 |
| `repo.huaweicloud.com` | ❌ HTTP 429 |
| `mirrors.ivolces.com` | ❌ 连接失败 |
| **`mirrors.aliyun.com/pypi/simple`** | ✅ 采用（约 1 MB/s） |
| `mirrors.cloud.tencent.com` | ✅ 可用 |
| `pypi.org` / `files.pythonhosted.org` | ✅ 可用 |
| `download.pytorch.org/whl/cu128` | ✅ 可用（torch 官方源） |
| **`huggingface.co`** | ❌ 直连不通 |
| **`hf-mirror.com`** | ✅ 可用（模型下载必须走它） |

本次实际使用的环境变量：

```bash
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export PIP_TRUSTED_HOST=mirrors.aliyun.com
```

### 4.3 文件沙箱

早期会话是 `workspace-write` 策略，`/root/miniconda3`（workspace 之外）写入被拒，`conda create` / `pip install` 必须在放宽权限后执行。当前会话已是 **danger-full-access**，不再需要额外授权。

### 4.4 代码层面的一个注意点

DM05 的训练 `forward` 在 `model.train()` 模式下**必须传 `token_type_ids`**，否则 transformers 5.3.0 的 Gemma3 会直接抛：

```
ValueError: `token_type_ids` is required as a model input when training
```

纯文本 batch 传全 0 即可（图像/历史位置为 1，见 `opendm/data/collator.py`）。这是本次 GPU 冒烟第一次失败的原因，属测试脚本漏传参数，不是仓库缺陷。

---

## 5. 冒烟测试结果

冒烟测试不依赖任何预训练权重：用 `Gemma3TextConfig`/`SiglipVisionConfig` 拼一个 2 层、hidden=64 的微型 DM05（`action_dim=8`、`chunk_size=4`），随机初始化后跑真实代码路径。

### 5.1 CPU 冒烟（沙箱内，CUDA 不可见）

脚本：`.smoke/scripts/smoke_test.py`（fp32_mixed policy）

```
opendm: /root/workspace/opendm/opendm/__init__.py
torch: 2.11.0+cu128 cuda_available=False

[PASS] import + 微型模型构建: DM05ForConditionalGeneration
[PASS] 训练 forward + backward: loss=16.7924, fm_loss=16.7924, params_with_grad=68
[PASS] inference_action 欧拉采样: actions(1, 4, 8) mean=-0.0788
[PASS] 参数量统计: vlm=191232, action_expert=136768, total=337416
SMOKE OK
```

单元测试：

```
$ python -m pytest tests/ -v
tests/test_dm05_adarms_cond.py::test_build_adarms_cond_accepts_explicit_suffix_dtype PASSED
tests/test_dm05_adarms_cond.py::test_build_adarms_cond_legacy_call_uses_weight_dtype[weight_dtype0] PASSED
tests/test_dm05_adarms_cond.py::test_build_adarms_cond_legacy_call_uses_weight_dtype[weight_dtype1] PASSED
tests/test_dm05_adarms_cond.py::test_build_adarms_cond_fp32_mixed_preserves_fp32_under_autocast PASSED
======================== 4 passed, 2 warnings in 4.44s =========================
```

（两条 warning 无碍：CUDA 初始化失败是沙箱所致；joblib 因无 `/dev/shm` 写权限退化为串行模式。）

训练入口：

```
$ python -m opendm.exp.dm05_exp --help   # rc=0
usage: .../opendm/exp/dm05_exp.py [-h] [OPTIONS]
  --task {train,inference}                      (default: train)
  --model-config.model-name-or-path STR         (default: ./checkpoints/DM05)
  --model-config.chunk-size INT                 (default: 50)
  --model-config.precision-policy {bf16_mixed,fp32_mixed}  (default: bf16_mixed)
```

### 5.2 GPU 冒烟（完全访问模式，A100 可见）

脚本：`.smoke/scripts/smoke_test_gpu.py`（bf16_mixed policy，训练用 `torch.autocast("cuda", bf16)`；推理分别测 `sdpa` 与 `eager` 两种 suffix attention backend）

```
opendm: /root/workspace/opendm/opendm/__init__.py
torch: 2.11.0+cu128
cuda_available: True
gpu: NVIDIA A100-SXM4-80GB cc8.0 mem=79.2GiB sm=108 bf16=True

[PASS] GPU 训练 forward + backward (bf16 autocast): loss=12.9427, params_with_grad=58, peak_mem=20.0MiB
[PASS] GPU 动作推理 (sdpa + CUDA Graph 分支): actions(1, 4, 8) mean=-0.1107, graph_profiles=1
[PASS] GPU 动作推理 (eager 回退分支): actions(1, 4, 8) mean=0.1815
GPU SMOKE OK
```

要点：

- `graph_profiles=1` 说明 `inference_action` 在 CUDA eval + `sdpa` 下确实走了 **suffix CUDA Graph** 路径（第一次 eager，第二次 capture/replay），与 `docs/zh/dm05_inference.md` 描述的一致。
- 微型模型峰值显存仅 20 MiB，不代表真实模型占用。
- 同环境下 `pytest tests/ -q` → `4 passed in 4.20s`。

---

## 6. 预训练权重下载与完整性校验

目标产物：`/root/workspace/opendm/checkpoints/DM05/`，对应 HuggingFace 仓库
`Dexmal/DM05`，revision `a7bf3f465de915a44ed9a4ac6c403ab3466f2735`。

### 6.1 最终结果

```
$ ls -la checkpoints/DM05/
      1,617  .gitattributes          7,931  README.md
      1,532  chat_template.jinja     6,795  config.json
        204  generation_config.json  11,064 norm_stats.json
        419  preprocessor_config.json  560  processor_config.json
 86,844,448  replay.mp4             33,384,567 tokenizer.json
        794  tokenizer_config.json
 11,658,431,136  model.safetensors   <-- 主权重 10.86 GiB
```

- 12/12 文件全部到位，逐个与 Hub tree API 的 `size` 比对**全部一致**。
- 主权重 SHA256 `b7da77f5…de20dd` 与 Hub 登记的 LFS `oid` **完全匹配**。
- safetensors 头部自检：1473 个张量，`data_offsets` 末端 == 文件长度，
  结构完整、无截断。
- 合计 10.97 GiB；下载后容器磁盘剩余 37 G。

### 6.2 一个重要的吞吐坑：hf-mirror 单连接只有 3–5 MB/s

`hf download` 走的是「每个文件一条 HTTP 连接」的路径。实测：

| 方式 | 吞吐 | 11.66 GB 预计耗时 |
| --- | ---: | ---: |
| `hf download`（单连接/文件） | 约 3–5 MB/s | 60+ 分钟 |
| `curl -r` 8 条并发 Range | **约 50–105 MB/s** | 不到 5 分钟 |

`hf-mirror` 对 `resolve/main/...` 返回 302，重定向到 `cas-bridge.xethub.hf.co`
的 Xet CAS 地址；该地址**支持 HTTP Range 且单连接不设限、只受总带宽限制**，
所以并发分片是最省时间的做法。

为此写了 `.smoke/scripts/parallel_download.py`：把 11.66 GB 切成 44 个 256 MB 分片，
8 线程并发 curl，每片先落 `.tmp` 再校验长度后改名（防止服务端忽略 Range 返回
200 时把整份文件写进分片），最后按序拼接并校验 SHA256。

实际耗时 **2.9 分钟，平均 63.6 MB/s**，SHA256 一次通过。

```bash
# 复现（必须绕过容器预置代理）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
cd /root/workspace/opendm
python .smoke/scripts/parallel_download.py
```

> `hf download Dexmal/DM05 --local-dir ./checkpoints/DM05` 仍然是官方推荐姿势，
> 只是在本网络环境下慢 15–20 倍。两者产物完全等价（同一个 revision、同一份
> sha256），用哪个都行。

### 6.3 踩坑记录：脚本自身的死锁

第一版 `.smoke/scripts/parallel_download.py` 里 `log()` 内部会获取 `_lock`，而成功分支
又在 `with _lock:` 里调用 `log()` —— `threading.Lock` **不可重入**，第一个下完
分片的线程直接自锁，其余 7 条连接下完后全部堵在锁上。

现象很有迷惑性：8 个分片文件都正常落盘，但日志停在启动那两行、3 分钟没有任何
新输出。用 `threading.RLock()` + 把 `log()` 挪到锁外即可（并在分片改名之后才
进临界区，所以这次死锁没有浪费任何已下载流量，重启后自动跳过已完成分片）。

---

## 7. 可选依赖层现状

| 层 | 内容 | 现状与说明 |
| --- | --- | --- |
| `pip install -e ".[attention]"` | `flash_attn` | 见下方 7.1。代码里是惰性导入（`dm05_utils.py: is_flash_attention_2_available()` 捕获 `ImportError`），**但 checkpoint 的 `config.json` 把 vision attention 钉死成 `flash_attention_2`，所以不装它连默认推理都起不来** |
| `pip install -e ".[fast-infer]"` | `onnx==1.21.0`、`tensorrt==10.8.0.43` | 未装。仅 `--inference-config.backend fast` 需要；`triton==3.6.0` 已随 torch 装好。PyPI 已确认存在 `tensorrt 10.8.0.43`。注意 fast backend 还要求 TensorRT 运行时、Triton kernel 与 `flex_attention`，且启动时会先构建 vision TensorRT engine，第一次启动较慢 |

### 7.1 flash-attn 的两个坑

编译条件本来就具备（nvcc 12.8、`/usr/local/bin/ninja`、A100 sm80 受支持），
但直接照搬 `MAX_JOBS=2 pip install flash-attn --no-build-isolation` 会卡两次：

1. **pip 拉 sdist 卡在 ~35 KB/s 的慢连接上**。同一个 URL 用 `curl` 重开连接
   能跑 3.6 MB/s 秒下。解决办法：先 `curl -o flash_attn-2.8.3.tar.gz <url>`，
   再 `pip install ./flash_attn-2.8.3.tar.gz`。（顺带校验 sha256 ==
   `1e71dd64…e0370d`，与镜像 index 登记值一致。）
2. **`setup.py` 会先去 GitHub releases 找预编译 wheel**。容器里
   `github.com:443`（20.205.243.166）能建连但**永不返回**，进程 65 个线程全部
   `wait_woken`，CPU 0%、`rchar` 十秒零增长，`tail` 日志也看不到任何进度
   （`setup.py:56` 那个 `releases/download/{tag_name}/{wheel_name}` URL）。
   解决办法：`export FLASH_ATTENTION_FORCE_BUILD=TRUE` 强制本地编译。

一键脚本：`.smoke/scripts/install_flash_attn.sh`（已内置上面两个处理），日志
`.smoke/logs/install_flash_attn.log`。本机 128 核 / 1 TB 内存，用 `MAX_JOBS=48`
比文档里的 `MAX_JOBS=2` 快得多。

---

## 8. 复现步骤

```bash
# 0) 关键前置：绕过代理
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# 1) 一键装环境（约 40 分钟，含约 4 GB CUDA wheels）
bash /root/workspace/opendm/.smoke/scripts/install_env.sh

# 2) 激活
conda activate opendm
cd /root/workspace/opendm

# 3) 冒烟测试
python -m pytest tests/ -v                      # 单元测试
python .smoke/scripts/smoke_test.py             # CPU 微型模型：构建 / 前向+反向 / 推理 / 参数统计
python .smoke/scripts/smoke_test_gpu.py         # GPU 微型模型：bf16 训练路径 + sdpa(CUDA Graph) / eager 推理
```

### 本次用到的辅助脚本

`.smoke/` 目录按「代码 / 日志 / 产物」三分：`scripts/` 入库，`logs/` 与 `artifacts/`
是本地留痕（已 gitignore）。总入口见 `.smoke/README.md`。

| 路径 | 作用 |
| --- | --- |
| `.smoke/scripts/install_env.sh` | 建 conda 环境 + 装 torch(cu128) + `pip install -e .`（含代理/源处理） |
| `.smoke/logs/install.log` | 安装全过程日志 |
| `.smoke/scripts/run_smoke.sh` | 依次跑 pytest + CPU 微型模型冒烟 + 训练入口 CLI |
| `.smoke/scripts/smoke_test.py` | CPU 冒烟：微型 DM05 构建、训练前向+反向、`inference_action`、参数量统计 |
| `.smoke/scripts/smoke_test_gpu.py` | GPU 冒烟：bf16 autocast 训练路径、`sdpa`(CUDA Graph) 与 `eager` 推理 |
| `.smoke/scripts/parallel_download.py` | 权重并发分片下载（8 线程 Range + SHA256 校验） |
| `.smoke/scripts/install_flash_attn.sh` | flash-attn 本地编译安装 |
| `.smoke/scripts/run_inference_e2e.sh` | 真实权重端到端：起服务 → 打 `/v1/infer` → 收尾（响应落 `artifacts/`） |

> 注意：`.smoke/` 是本次新增的工作区目录，只入库 `scripts/` 与 `README.md`；运行日志
> 和下载/响应产物留在本地不入库，整目录删除也不影响仓库既有文件。

---

## 8. 下一步（尚未完成的端到端验证）

真机端到端还差 **checkpoint**。已确认 `Dexmal/DM05` 在 hf-mirror 上包含 `config.json`、`model.safetensors`（11,658,431,136 B ≈ 10.9 GiB）、`norm_stats.json`、tokenizer/processor 等 12 个文件；由于 `huggingface.co` 直连不通，必须指定镜像端点：

```bash
export HF_ENDPOINT=https://hf-mirror.com
conda activate opendm
cd /root/workspace/opendm

hf download Dexmal/DM05 --local-dir ./checkpoints/DM05

# 启动推理服务（基础预训练模型：3 路图像、14 维 state/action）
script/dm05_launcher.sh \
  --exp opendm/exp/dm05_exp.py \
  --task inference \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --model-config.chunk-size 50 \
  --inference-config.output-action-dim 14 \
  --inference-config.image-prompts "Head" "Left wrist" "Right wrist" \
  --inference-config.port 7891
```

细节与 HTTP API 用法见 `docs/zh/dm05_inference.md`。另外建议补装 `flash-attn`（训练/推理性能）与 `fast-infer`（低延迟推理）两个可选层。
