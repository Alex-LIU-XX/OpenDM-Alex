# DM0.5 当前权重与 Benchmark 对应关系梳理

- 记录时间：2026-09-21 12:45 (CST)
- 机器：单卡 NVIDIA A100-SXM4-80GB，容器内 root
- 仓库：`/root/workspace/opendm`，commit `fbab441`
- 本地权重：`./checkpoints/DM05`（HuggingFace `Dexmal/DM05`，revision `a7bf3f465de915a44ed9a4ac6c403ab3466f2735`）
- 文档性质：**面向"想现在拿这份权重跑测试"的排查记录**。第 3、4、5 节的事实来自仓库内文档与 README，第 1、2、7 节为本次对权重文件/环境实测核对的结果；来源与出处逐条标注，未标注即本机实测。
- 相关文档：[DM0.5 技术博客整理](DM0.5_tech_blog_notes.md)、[环境配置与冒烟测试记录](opendm_env_setup_and_smoke_test.md)

---

## 0. 结论速览

| 问题 | 结论 |
| --- | --- |
| 当前权重是哪个 benchmark 的？ | **不是任何一个 benchmark 的权重**，它是 DM0.5 的**基础预训练权重**（4B VLM + 680M Action Expert），定位是"零样本真机 + SFT 基座" |
| 能直接拿它刷公开榜单吗？ | **不能**。仓库里每个 benchmark 都是"先用它做 SFT → 再用微调后的 checkpoint 起服务 → 评测客户端通过 HTTP 打推理服务"的三段式，且文档明确禁止跨 benchmark 混用 checkpoint / 入口 / 维度 |
| 它自带什么规格？ | 3 路图像（Head / Left wrist / Right wrist）、14 维 state/action、`chunk_size=50`、`norm_stats.json` 含 `DOS W1`（默认）与 `Aloha` 两个 profile |
| 仓库支持哪些 benchmark？ | 仿真 4 个：LIBERO、RoboTwin 2.0、VLA-Arena、RoboDojo-Sim；真机 2 个：RoboChallenge Table30 v2、SO101 Pick Cube |
| 最便宜的"跑测试"是什么？ | 不碰 benchmark：用当前权重起推理服务 + `bash tests/curl_demo.sh`，验证 3 图/14 维/chunk50 链路（已于 2026-09-21 验证通过） |
| 最适合的真 benchmark 是哪个？ | **LIBERO**（单臂仿真、生态成熟、有 LoRA 入口、有官方 Docker 评测客户端）；RoboTwin 2.0 次之（有官方榜单数值可对标） |
| 本机现在能直接跑吗？ | 不能，有三道坎：① `workspace-write` 沙箱下 CUDA/NVML 不可见；② 根分区只剩 37 G，而各 benchmark 数据包 13–37 GiB；③ 评测建议 ≥2 卡，本机 1 卡 |

---

## 1. 当前权重到底是什么

### 1.1 文件清单（本机 `ls -la checkpoints/DM05/`）

| 文件 | 大小 | 用途 |
| --- | ---: | --- |
| `model.safetensors` | 11,658,431,136 B ≈ 10.86 GiB | 主权重（1473 个张量，SHA256 `b7da77f5…de20dd` == Hub LFS oid） |
| `config.json` | 6,795 B | 模型结构配置 |
| `norm_stats.json` | 11,064 B | 归一化统计（**决定能对接哪些机器人 profile**） |
| `tokenizer.json` / `tokenizer_config.json` / `chat_template.jinja` | 33 MB / 794 B / 1,532 B | 分词器与对话模板 |
| `preprocessor_config.json` / `processor_config.json` | 419 B / 560 B | 图像预处理 |
| `generation_config.json` | 204 B | 生成配置 |
| `replay.mp4` | 86,844,448 B | 演示回放视频 |
| `README.md` / `.gitattributes` | 7,931 B / 1,617 B | 模型卡 / LFS 规则 |

### 1.2 结构关键值（本机读 `config.json` 核对）

| 项目 | 值 |
| --- | --- |
| `architectures` | `DM05ForConditionalGeneration` |
| `model_type` | `dm05` |
| `chunk_size` | **50** |
| `action_dim` | **32**（模型内部动作维度） |
| `dtype` | `bfloat16` |
| `transformers_version` | `5.3.0` |
| Action Expert（`action_config`） | `hidden_size=1024`、34 层、8 heads、`head_dim=256`、`vocab_size=262208` |
| VLM 主干 | Gemma 3 4B（详见 [技术博客整理](DM0.5_tech_blog_notes.md) 第 11 节） |

### 1.3 `norm_stats.json` 决定"能对接谁"（本机 dump 结果）

```
default_robot_type = DOS W1
norm_stats/（默认 profile）: action/state 各 14 维（q01/q99/mean/std）
norm_stats_by_robot/DOS W1   : 14 维
norm_stats_by_robot/Aloha    : 14 维
```

也就是说：**这份基础权重自带的是 `DOS W1`（Dexmal 自研双臂移动操作机器人）与 `Aloha` 两个真机 profile，14 维 state/action**。这一点与推理文档一致（`docs/zh/dm05_inference.md:54-58,100-106`）：

> 基础预训练模型：入口 `opendm/exp/dm05_exp.py`，checkpoint `Dexmal/DM05`，chunk 50，3 图，14 维 action。
> 将 `observation.robot_type` 设为 `DOS W1` 或 `Aloha` 可选择归一化 profile；省略时用默认 `DOS W1`。
> 直接使用基础预训练模型时，应显式提供 `observation.control_mode` 和 `observation.speed`（`speed` 默认 `"0.5"`）。

`control_mode` / `speed` 这两个**文本条件**是基础预训练权重特有的输入要求（`docs/zh/dm05_inference.md:337,395`），SFT checkpoint 只有在训练数据里带这两个字段时才需要——这从侧面说明：**这份权重是按"真机多本体预训练"的条件训练的，不是为某个仿真榜单训练的**。

### 1.4 官方模型卡里的定位

`checkpoints/DM05/README.md` 中：

- `tags`：`robotics`、`robot-control`、`vision-language-action`、`vla`、`dm05`、`dm0.5`、`opendm`
- 单独列出的 **Fine-Tuned Models** 表（共 6 个）才是"面向评测"的权重：

| 微调权重 | 面向的 benchmark |
| --- | --- |
| `Dexmal/DM05-libero` | LIBERO |
| `Dexmal/DM05-robotwin2` | RoboTwin 2.0 |
| `Dexmal/DM05-SO101-Pick-Cube` | SO101 真机 Pick Cube |
| `Dexmal/DM05-Vla-Arena` | VLA-Arena |
| `Dexmal/DM05-Table30v2`（collection） | RoboChallenge Table 30 v2 真机 |
| `Dexmal/DM05-MEM-Robodojo-Sim` | RoboDojo-Sim（ARX X5 双臂） |

**基础权重自身没有出现在这张表里**——这就是"当前权重不对应单一 benchmark"的直接证据。

---

## 2. 为什么"权重 → benchmark"不是一一对应

仓库的评测范式是固定三段式（`docs/zh/dm05_inference.md` 第 2 节、各 benchmark 文档）：

```
Dexmal/DM05（基础权重）
   └─ ① SFT / LoRA 微调（该 benchmark 的数据 + 该 benchmark 的 playground 入口）
        └─ ② 起推理服务（同一入口 + 微调后的 checkpoint + 匹配的 chunk/image/dim/norm_stats）
             └─ ③ benchmark 客户端（Docker 镜像或官方仓库）通过 HTTP 请求动作预测
```

`docs/zh/dm05_inference.md:52-63` 的"选择推理入口"表把这条规则写死了：

| 使用场景 | 入口 | 常用 checkpoint | Chunk size | 图片数 | Action 维度 |
| --- | --- | --- | ---: | ---: | ---: |
| 基础预训练模型 | `opendm/exp/dm05_exp.py` | `Dexmal/DM05` | 50 | 3 | 14 |
| LIBERO | `playground/dm05_libero.py` | `Dexmal/DM05-libero` | 10 | 2 | 7 |
| RoboTwin 2.0 | `playground/dm05_robotwin2.py` | `Dexmal/DM05-robotwin2` | 50 | 3 | 14 |
| DM05-MEM | `playground/dm05_mem_sft_demo.py` | `Dexmal/DM05-MEM` | 50 | 3 | 14 |
| Demo 或自定义 SFT | `playground/dm05_sft_demo.py` 或自定义入口 | SFT checkpoint | 训练值 | 训练值 | 训练值 |
| LIBERO LoRA | `playground/dm05_libero_lora.py` | LIBERO LoRA step checkpoint | 10 | 2 | 7 |

> 原文（`docs/zh/dm05_inference.md:63`）：**"不要混用不同 benchmark 的 checkpoint、入口或推理维度。"**

配套约束还有两条容易踩：

- **归一化统计必须同源**：推理优先读 checkpoint 目录下的 `norm_stats.json`，找不到才回退 `./norm_stats/`；`norm_stats_by_robot` 里多 profile 时按 `observation.robot_type` **精确匹配**，未知机型不会静默回退默认 profile。
- **大改维度就得重训**：`--inference-config.output-action-dim` 必须与归一化向量维度一致；图片数量/顺序必须与 `--inference-config.image-prompts` 一致。

---

## 3. Benchmark 全景表（仓库支持范围）

| Benchmark | 类型 | 训练/推理入口 | 官方微调权重 | 规格（chunk/图/维度/机型） | 评测客户端 | 参考指标出处 |
| --- | --- | --- | --- | --- | --- | --- |
| **LIBERO** | 单臂仿真 | `playground/dm05_libero.py` | `Dexmal/DM05-libero` | 10 / 2 / 7 / `Franka` | `dexbotic-benchmark` + LIBERO 子模块，Docker 镜像 `dexmal/dexbotic_benchmark` | README 榜单 SR 99.0% |
| **RoboTwin 2.0** | 双臂仿真 | `playground/dm05_robotwin2.py` | `Dexmal/DM05-robotwin2` | 50 / 3 / 14 / `Aloha RoboTwin2` | `dexbotic-benchmark` + RoboTwin 子模块 | README 榜单 Clean 93.6 / Rand 93.3 |
| **VLA-Arena** | 仿真（170 任务 = 11 suite × 3 level） | `playground/dm05_vla_arena.py` | `Dexmal/DM05-Vla-Arena` | 7 维 action（文档命令） | `PKU-Alignment/VLA-Arena` + 仓库 `third_party/vla_arena/{eval.py,eval_config.yaml}` | README 榜单 L0 89.0 / L1 53.6 / L2 44.1 |
| **RoboDojo-Sim** | 双臂仿真（ARX X5） | `playground/dm05_mem_sft_demo.py`（走 DM05-MEM 基座） | `Dexmal/DM05-MEM-Robodojo-Sim` | 3 图 / 14 维 / 历史帧 20 | 官方评测 YAML（XPolicyLab 集成） | README 榜单 Score 24.90 / SR 19.34% |
| **RoboChallenge Table30 v2** | **真机平台** | `third_party/robochallenge_inference/execute.py` | `Dexmal/DM05-Table30v2`（collection） | arx5 / ur5 / aloha / w1 四套配置 | RoboChallenge 平台（需账号与 submission） | README 榜单 Score 54.42 / SR 43.0% |
| **SO101 Pick Cube** | **真机** | `playground/dm05_so101_lora.py` | `Dexmal/DM05-SO101-Pick-Cube` | 50 / 2 / 6 / LoRA | 真机 | 文档未给公开数值 |
| （基础权重自身） | **真机零样本** | `opendm/exp/dm05_exp.py` | `Dexmal/DM05` | 50 / 3 / 14 / `DOS W1` 或 `Aloha` | 无公开 harness（博客自建真机实验） | 博客 Table 1/2（Franka 与 Dexmal-Mirror），数值未公开 |

---

## 4. 各 benchmark 详解

### 4.1 LIBERO（推荐首选）

- **数据**：`Dexmal/libero`（HF dataset），**实测 11 个文件、34.85 GiB**（9 个 4 GiB 分片 `libero.tar.part-000…008`）。
- **数据准备**：`script/libero_runner.sh dataset` → 下载到 `./data/.hf_downloads/libero`，自动合并分片、解压、整理到 `./data/libero`；子集含 `libero_pi0_all`、`libero_10`、`libero_goal`、`libero_object`、`libero_spatial`。
- **训练**（`docs/zh/dm05_libero.md`）：

  ```bash
  script/dm05_launcher.sh \
    --exp playground/dm05_libero.py \
    --task train \
    --nproc_per_node 8 \
    --data-config.dataset-name libero_pi0_all \
    --model-config.model-name-or-path ./checkpoints/DM05 \
    --model-config.chunk-size 10 \
    --trainer-config.num-train-steps 100000
  ```

  参考配置是 **8 卡 / 100k 步**；LoRA 路线（`docs/zh/dm05_libero_lora_training.md`）参考配置为 **8×RTX 4090D、batch 32、50k 步、rank 32、LR 5e-4**，其 49k checkpoint 的 LIBERO 总体成功率 **98.30%**——这是目前仓库里唯一给出"微调代价 + 结果"的完整参考，很适合用来估算单卡 A100 的时间成本。
- **评测**：`script/dm05_launcher.sh --task inference --exp playground/dm05_libero.py --model-config.chunk-size 10 --inference-config.output-action-dim 7 --inference-config.image-prompts "Head" "Left wrist"`；客户端 `git clone https://github.com/dexmal/dexbotic-benchmark.git` → `git submodule update --init --recursive libero` → `docker pull dexmal/dexbotic_benchmark`；配置 `evaluation/configs/libero/example_dm05_libero.yaml`，`benchmark` 可选 `libero_spatial` / `libero_goal` / `libero_object` / `libero_10`；结果在 `output_dir` 下的 `results.json` 与 `logs/evaluation.log`。
- **门槛**：**建议 ≥2 卡**（一卡推理服务、一卡仿真客户端）；需要 Docker。

### 4.2 RoboTwin 2.0

- **数据**：`Dexmal/robotwin2-full`，**实测 6 个文件、36.69 GiB**（`robotwin2.tar.part-aa…ad`）。解压后为 `data/robotwin2.0/{jsonl,video}/<task>/{clean,randomized}`。
- **注册数据集**：`opendm/dataset/robotwin2.py` 中的 `robotwin2_generalist`。
- **参考指标**（`docs/zh/dm05_robotwin2.md` 开头表）：Clean **93.6** / Randomized **93.3** / Average **93.5**（与 README 榜单一致）。
- **评测**：同样用 `dexbotic-benchmark`（`git submodule update --init --recursive RoboTwin` + 同一 Docker 镜像），配置 `evaluation/configs/robotwin2/adjust_bottle.yaml`；`task_name` 从 50 个任务中逐个评测，`task_config` 决定 `demo_clean` 或 `demo_randomized`，`action_horizon=50`。
- **额外门槛**：RoboTwin 还需自行下载 **assets、object-data texture library、embodiment 文件**（官方安装文档），依赖比 LIBERO 更重。

### 4.3 VLA-Arena

- **数据**：`Dexmal/vla_arena_L0_L`，**实测 6 个文件、35.04 GiB**。
- **评测规模**：**170 个任务 = 11 个 suite × 3 个 level**；配置里 `task_suite_name` 可设 `"all"` 或单个 suite（如 `safety_static_obstacles`），结果按 `seed_<N>/results_<timestamp>.json` 保存（成功率 + cost）。
- **客户端**：克隆 `https://github.com/PKU-Alignment/VLA-Arena`，按文档 `apt-get install libosmesa6-dev libglfw3 libgl1-mesa-glx libglib2.0-0`、`pip install robosuite==1.5.1 bddl numpy==1.26.4 …`、`export MUJOCO_GL=osmesa`，再把 `third_party/vla_arena/eval.py` 与 `eval_config.yaml` 复制进 `vla_arena/models/DM05/`。
- **门槛**：客户端依赖最杂（robosuite/BDDL/MuJoCo 渲染），适合作为第二、第三选择。

### 4.4 RoboDojo-Sim（注意：不是用当前权重直接微调）

- 走的是 **`Dexmal/DM05-MEM`** 基座而不是 `Dexmal/DM05`（`docs/zh/dm05_robodojo.md`）。
- 数据：`Dexmal/robodojo-sim`，**实测 4 个文件、13.14 GiB**（本表中最小的一个）。
- 文档明确说明：README 榜单数值对应已发布的 generalist **`DM05-MEM-Robodojo-Sim`**；文档里的 `cover_blocks` 单任务 SFT 流程**不能复现榜单分数**。
- 归一化统计需要单独从 `DM05-MEM-Robodojo-Sim/norm_stats.json` 下载，否则复现效果会打折。

### 4.5 RoboChallenge Table 30 v2（真机平台，本机不可行）

- 客户端在 `third_party/robochallenge_inference/`：连平台 → 取 submission 中的 job → 拉机器人观测 → 调 DM05 policy → 回传动作。
- 支持四种机型：`arx5`、`ur5`、`aloha`、`w1`；TensorRT vision engine 按图片数区分（`h8`：ARX5 用 3 当前图 + 5 历史 slot；`h2`：UR5 用 2 图；`h3`：ALOHA / W1 用 3 图），首次 fast backend 启动会自动构建。
- Runtime 默认值：ARX5 `action_horizon=50` / playback 25；UR5 `action_horizon=25`；ALOHA / W1 默认 25。
- 需要平台账号 + 真机，本机无法复现，只能作为"了解榜单来源"。

### 4.6 SO101 Pick Cube（真机 LoRA）

- 数据：`Dexmal/so101_pick_cube`，注册于 `opendm/dataset/so101.py`；规格 **2 图（Head / Left wrist）、6 维 action、relative action mode、chunk 50、含 state**。
- LoRA 参考配置：8 卡、单卡 batch 8、10,000 步、LR 1e-4、`all-linear`、VLM/AE 开 gradient checkpointing。
- 需要 SO101 真机（`docs/zh/robot_platforms.md` 是硬件改装指南）。

---

## 5. 官方榜单（README "Benchmark Results"）

列顺序为 README 原表：Benchmark / Metric / **DM0.5** / Pi0 / Pi0.5 / GROOT-N1.7。

| Benchmark | Metric | DM0.5 | Pi0 | Pi0.5 | GROOT-N1.7 |
| --- | --- | ---: | ---: | ---: | ---: |
| LIBERO | SR | **99.0%** | 94.4% | 96.9% | 97.0% |
| RoboTwin 2.0 | Clean | **93.6%** | 65.9% | 82.7% | - |
| RoboTwin 2.0 | Rand | **93.3%** | 58.4% | 76.8% | - |
| VLA-Arena | L0 | **89.0%** | 82.3% | 64.3% | - |
| VLA-Arena | L1 | **53.6%** | 32.2% | 35.6% | - |
| VLA-Arena | L2 | **44.1%** | 11.4% | 24.5% | - |
| RoboDojo-Sim | Score / SR | **24.90 / 19.34%** | 3.48 / 1.53% | 11.41 / 6.91% | 2.85 / 1.31% |
| RoboChallenge Table30V2 | Score / SR | **54.42 / 43.0%** | - | 31.48 / 14.3% | - |

> 表下的官方注解：RoboDojo-Sim 的榜单数值属于已发布的 generalist `DM05-MEM-Robodojo-Sim`；其链接的指南是 `cover_blocks` 单任务 SFT 参考，**训练设置不能复现表内分数**。
>
> 另外，博客（见 [技术博客整理](DM0.5_tech_blog_notes.md) 第 10 节）里的零样本数据（8 个操作原语 × 7 类语义约束，Franka 平台 `Pi0.5-Droid` vs `DM0.5-Droid`、Dexmal-Mirror 平台 `DM0` vs `DM0.5`）是**真机自建评测**，仓库内没有对应 harness，数值以图表呈现、正文未给完整数字。

**关键提醒**：这张榜单是"DM0.5 这个模型在各自 benchmark 上微调后的成绩"，**不是** `checkpoints/DM05` 这份基础权重能直接跑出来的数字。用基础权重去跑 LIBERO（2 图/7 维/`Franka`）会直接因维度、归一化统计与入口不匹配而失败。

---

## 6. 不碰 benchmark 也能做的两件事

1. **纯推理冒烟**（已在本仓库验证通过，见 [环境记录](opendm_env_setup_and_smoke_test.md) 第 6 节）：

   ```bash
   script/dm05_launcher.sh \
     --exp opendm/exp/dm05_exp.py \
     --task inference \
     --model-config.model-name-or-path ./checkpoints/DM05 \
     --model-config.chunk-size 50 \
     --inference-config.output-action-dim 14 \
     --inference-config.image-prompts "Head" "Left wrist" "Right wrist" \
     --inference-config.port 7891

   bash tests/curl_demo.sh http://127.0.0.1:7891/v1/infer      # legacy 接口：/process_frame
   ```

   基础权重还需显式带 `control_mode` 与 `speed`（默认 `"0.5"`）。产出是 `50 × 14` 的 action chunk + `latency_ms`，只证明**链路**通，不产生任何 benchmark 指标。
   - 注意：checkpoint 的 `config.json` 把 vision attention 钉成 `flash_attention_2`，**不装 `flash-attn` 连默认推理都起不来**（见环境记录第 7 节）。

2. **Demo SFT 链路验证**（`docs/zh/dm05_finetuning.md`）：用仓库内置 `assets/demo` 数据 + `playground/dm05_sft_demo.py` 走完"训练 → 保存 → 推理 → 服务校验"，image keys 为 `images_1/2/3`、`output_action_dim=14`、`chunk_size=50`，与基础权重规格一致。这是**在自己的数据上做 SFT 之前**最划算的一次排练。

---

## 7. 本机可行性评估（2026-09-21 12:4x 实测）

### 7.1 数据集体量 vs 磁盘

| 数据源 | 实测体量 | 说明 |
| --- | ---: | --- |
| `Dexmal/libero` | 34.85 GiB | 9 × 4 GiB 分片 |
| `Dexmal/robotwin2-full` | 36.69 GiB | 4 个分片（10 GiB × 3 + 6.7 GiB） |
| `Dexmal/vla_arena_L0_L` | 35.04 GiB | 4 个分片 |
| `Dexmal/robodojo-sim` | 13.14 GiB | 2 个分片（本表最小） |

根分区现状（`df -h /`）：**98 G 总 / 62 G 已用 / 37 G 可用（63%）**。其中：

```
43G  /root/miniconda3      （conda 环境）
17G  /root/.cache          （pip 等缓存，可直接清理）
11G  ./checkpoints         （DM05 权重 + 验证副本）
```

`script/libero_runner.sh` 的流程是"下载分片 → 解压到 `.extracted` → 整理到 `data/libero`"，**下载目录的分片不会自动删除**，峰值占用可能达到数据集本体的 2–3 倍。结论：**跑任何 benchmark 前必须先腾出至少 60–80 G**，最直接的是清 `/root/.cache`（+17 G），必要时删 `checkpoints/DM05-sdpa-verify` 等验证副本。

### 7.2 GPU 与沙箱

```
$ nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
Failed to initialize NVML: Unknown Error
```

- 这与 [环境记录](opendm_env_setup_and_smoke_test.md) 第 2 节的结论完全一致：**`workspace-write` 沙箱拦住 `/dev/nvidia*`，CUDA 不可见（NVML 初始化失败）**；切到完全访问模式后同一条命令正常。
- 所以：**当前沙箱下任何 GPU 训练/推理都会失败**，跑之前必须先放宽权限。

### 7.3 卡数

- 物理机是 **1 张 A100-SXM4-80GB**（另有同机其他容器占用的约 7.7 GiB，与本环境无关）。
- LIBERO / RoboTwin 文档都建议 **≥2 卡**（一卡推理服务、一卡仿真客户端）。单卡可以"分时复用"（先起服务、再跑客户端，或服务与客户端抢同一张卡），但吞吐会明显下降。
- 训练侧：单卡 A100-80G 跑 Full SFT 会非常慢（参考配置是 8 卡 × 100k 步），**优先 LoRA**（LIBERO 有现成 LoRA 入口，参考 8×4090D / 50k 步）。

---

## 8. 建议的可执行路线（按代价从低到高）

| 路线 | 内容 | 需要的资源 | 产出 |
| --- | --- | --- | --- |
| **A. 纯冒烟** | 基础权重起服务 + `curl_demo.sh` | 放宽沙箱、1 卡、无额外磁盘 | 证明链路通（50×14 chunk） |
| **B. Demo SFT 排练** | `assets/demo` + `playground/dm05_sft_demo.py` 走完训练→保存→推理 | 放宽沙箱、1 卡、几 GB 磁盘 | 一份可复用的 SFT 流程与脚本 |
| **C. LIBERO 真评测** | 清磁盘 → 下 `Dexmal/libero` → LoRA SFT → 起服务 → `dexbotic-benchmark` Docker 客户端 | ≈35 G+ 磁盘、1–2 卡、Docker | LIBERO 四个子集的成功率（可对标 99.0% / 98.30%） |
| **D. RoboTwin 2.0 真评测** | 同上 + 装 RoboTwin 模拟器 + 下载 assets/texture/embodiment | ≈37 G+ 磁盘、2 卡、Docker | 可对标 Clean 93.6 / Rand 93.3 |
| **E. VLA-Arena** | 克隆 `PKU-Alignment/VLA-Arena` + MuJoCo 依赖 + 复制 `third_party/vla_arena` 配置 | ≈35 G+ 磁盘、1–2 卡 | 170 任务的成功率与 cost |
| **F. 真机（Table30v2 / SO101）** | 平台账号 / 真机硬件 | 不可在本机复现 | 榜单数值来源说明 |

---

## 9. 本次核对用的命令（可复现）

```bash
cd /root/workspace/opendm

# 1) 权重结构与归一化 profile
python -c "
import json
c=json.load(open('checkpoints/DM05/config.json'))
print(c['chunk_size'], c['action_dim'], c['dtype'])
n=json.load(open('checkpoints/DM05/norm_stats.json'))
print('default_robot_type:', n['default_robot_type'])
print('profiles:', list(n['norm_stats_by_robot'].keys()))
print('action dim:', len(n['norm_stats']['action']['mean']))
"

# 2) benchmark 数据集体量（必须绕过容器代理）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
curl -s "https://hf-mirror.com/api/datasets/Dexmal/libero?blobs=true" | python -c "
import json,sys; d=json.load(sys.stdin)
print(round(sum(s.get('size',0) or 0 for s in d['siblings'])/2**30, 2), 'GiB')"

# 3) 环境现状
df -h / ; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv

# 4) benchmark 入口与规格（文档出处）
sed -n '52,63p' docs/zh/dm05_inference.md
```

---

## 10. 待确认的问题

1. **目标 benchmark 选哪个？** 取决于目的：要"最容易跑通"选 LIBERO；要"能对上公开榜单数值"选 RoboTwin 2.0 或 LIBERO；要"复现真机成绩"需要真机，本机不可行。
2. **是否接受先做 SFT？** 当前权重无法直接产出任何 benchmark 指标；如果只想验证权重本身，走路线 A/B。
3. **磁盘怎么腾？** 建议先清 `/root/.cache`（≈17 G），再决定是否需要删除验证副本 `checkpoints/DM05-sdpa-verify`。
4. **是否放宽沙箱？** 当前 `workspace-write` 下 CUDA 不可见，任何 GPU 步骤都需切到完全访问模式。

---

> 附：本文档中标注"实测"的数值均为 2026-09-21 12:4x 在本机执行的结果；其余出处已标到具体文档小节/行号。若后续下载或微调产生了新 checkpoint，请以对应 benchmark 的 playground 入口 + 该 checkpoint 的 `norm_stats.json` 为准，**不要跨 benchmark 复用维度与统计**。
