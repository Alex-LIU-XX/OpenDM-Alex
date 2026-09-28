# DM0.5 四项设计在 OpenDM 开源代码中的落地核查报告

- 核查时间：2026-09-28 10:29 (CST)
- 核查对象：`/root/workspace/opendm`（`dev/Alex`，HEAD = `122d4ec`，工作区 clean）
- 对照基线：`docs/Alex/DM0.5_tech_blog_notes.md` 第 3–6 节（DM0.5 官方技术博客的方法描述）
- 对比对象：`/root/workspace/starvla/starVLA/model/framework/VLM4A/PI05.py`（StarVLA 的 π0.5 实现）
- 文档性质：**独立复核**（非转述既有笔记）。所有判定均给出可点击核对的文件:行号。

---

## 0. 结论速览

| # | 方法组件 | 开源代码实现度 | 一句话判定 |
| --- | --- | --- | --- |
| 1 | **Context Abstraction Layer** | 🟡 **主干已实现，训练策略只做了一半** | 历史 slot 注入、`<unused0>`/`<unused1>` 占位、16 token 空间池化全部具备；但时间采样是**固定 1 FPS 均匀网格**，且**没有随机历史长度、没有专门的历史增强** |
| 2 | **Embodiment CoT Tasks**（11 个自回归任务） | ❌ **完全未实现** | 训练 loss 只有 flow matching（`dm05_arch.py:917`），输出 `logits=None`；全历史 40 个 commit 检索 CoT/DP/monotonic 关键字零命中；数据 schema 连对应字段都没有 |
| 3 | **Trajectory Alignment Layer** | ❌ **完全未实现** | `BuildActionChunk` 是**固定时间索引切片 + 末尾重复填充**，`action_mask` 恒为全 1，没有任何锚点匹配 / 单调约束 / 动态规划 |
| 4 | **数据清洗流水线** | ❌ **基本未实现** | 只有"丢空行 + 丢每 episode 最后一帧"；五条策略一条都没有，最接近的仅是归一化时的 q01/q99 数值裁剪 |

> 仓库既有笔记 `docs/Alex/DM0.5_tech_blog_notes.md:246-263` 得出方向一致的结论。本报告为独立复核，并补齐了它未展开的细节：历史窗口实际多长、时间采样是否随机、`action_mask` 为何恒为 1、历史缺失时如何退化、训练/推理槽位数是否一致等。
>
> **一句话**：开源版 = **架构 + 历史上下文融合 + Flow Matching 训练/推理加速**；**CoT 监督、动态动作匹配、数据清洗**这三条训练侧/数据侧设计**未随代码开源**。

---

## 1. Context Abstraction Layer —— 🟡 部分实现

### 1.1 已实现的部分（架构 / 数据 / 推理三段全通）

| 博客机制 | 代码落地 | 位置 |
| --- | --- | --- |
| 历史帧 slot | `LoadHistory`：从**主视角** `image_keys[0]` 按 slot 回溯 | `opendm/data/transforms.py:229-268` |
| 压缩为定长 visual token | 每张历史图 = `HISTORY_TOKENS_PER_IMAGE = 16` 个软 token；空间上 `adaptive_avg_pool2d(4×4)` 把 256 个 patch 特征池化为 16 | `opendm/constants/robot.py:2-7`；`dm05_arch.py:1468-1478` |
| 历史注入方式 | `<unused0>` 占位 → `masked_scatter` 填入池化视觉特征；`token_type_ids` 置 1 | `transforms.py:748-759`、`transforms.py:834-838`、`dm05_arch.py:1481-1486` |
| 历史缺失的优雅退化 | `<unused1>`（id=7）占位：embedding 置 0、**注意力中屏蔽**、position_id 不计入 | `dm05_utils.py:251-272`、`dm05_arch.py:1455-1457`、`dm05_arch.py:1490-1494` |
| batch 打包 | `history_mask` / `history_pixel_values` 逐样本 padding 后 concat | `opendm/data/collator.py:93-120` |
| 推理侧（含 TRT 快路径） | current + 补零历史拼成定长 vision batch，历史特征池化到 16 token | `opendm/infer/dm05_trt_utils.py:25,28,50`；`opendm/infer/dm05_infer.py:345-385` |

核心代码（`dm05_arch.py:1468-1494`，有删节）：

```python
spatial = int(image_features.shape[1] ** 0.5)
grid = image_features.view(-1, spatial, spatial, hidden).permute(0, 3, 1, 2)
grid = F.adaptive_avg_pool2d(grid, output_size=(HISTORY_POOL_SIZE, HISTORY_POOL_SIZE))
image_features = grid.permute(0, 2, 3, 1).reshape(-1, HISTORY_POOL_SIZE**2, hidden)
history_mask_expanded = history_mask.unsqueeze(-1).expand_as(inputs_embeds)
prefix_inputs_embeds = inputs_embeds.masked_scatter(history_mask_expanded, image_features)
```

### 1.2 三个关键缺口（与博客描述存在实质差异）

**① 时间采样是"固定均匀"，不是"采样策略"**

```python
# opendm/data/transforms.py:242,255-256
uniform_fps: float = 1.0
...
for slot in range(self.max_history_images, 0, -1):
    raw_index = frame_index - int(round(slot / self.uniform_fps * source_fps))
```

- 回溯位置严格确定，无抖动、无自适应、无多尺度。
- **窗口最长 32 秒**（32 slot × 1 FPS，`transforms.py:242` 默认 `max_history_images=32`），而博客表述为"最长约 60 秒"。要拿到 60 秒需把 `max_history_images` 设为 60，仓库中没有任何地方这么配。
- 仓库文档亦如此描述：`docs/en/dm05_mem_sft.md:18,29`（"32 main-view history slots at 1 FPS"）。

**② 没有 randomized history lengths**

- 全仓 `random` 只出现在 `opendm/data/augmentations.py`（图像增强）和 `opendm/exp/dm05_exp.py:946`（`np.random.seed`）。
- 历史长度是由 `frame_index` 决定的**确定性值**，训练时不随机采样。
- 唯一的"变长"来自 episode 开头帧数不足 → 自然产生 `<unused1>` pad。配合 pad masking，模型**确实能**处理长短不一乃至全空的历史（能力具备），但博客强调的是**主动随机化训练**，这一点没有。

**③ 没有"历史增强"，只有与当前帧共用同一套图像增强**

```python
# opendm/data/transforms.py:76-81
def __call__(self, data):
    data["images"] = self._transform_images(data["images"])
    history_images = data.get("history_images")
    if history_images:
        data["history_images"] = self._transform_images(history_images)
    return data
```

`TrainingTransformPipeline`（PadToSquare / Resize / RandomResizedCrop(0.95) / Rotate(±2) / ColorJitter）对两者一视同仁。这是通用图像增强，**不是**针对历史帧的增强策略。

### 1.3 两个附加坑

**④ 训练/推理默认槽位数不一致**

| 位置 | 槽位数 | 说明 |
| --- | --- | --- |
| 训练 `ChatTokenization` | **32** | `dm05_exp.py:367-374` 未传参 → 默认 32 |
| 推理 `DM05InferenceConfig.max_history_images` | **5** | `dm05_exp.py:510` |
| MEM playground 推理 | **32** | `playground/dm05_mem_sft_demo.py:82` 覆盖 |

占位符长度按 `n_valid` 动态生成，因此功能上不会崩，但**服务默认只喂 5 帧历史**。

**⑤ 基础训练入口默认不开历史**

`DM05DataConfig.is_history = False`（`dm05_exp.py:290`），仅 `demo_mem` / RoboDojo MEM 两个 playground 打开。也就是说普通 SFT 走的是**无历史路径**。

---

## 2. Embodiment CoT Tasks（11 个自回归任务）—— ❌ 完全未实现

### 2.1 证据链

**① 训练 loss 只有一项，且无文本监督**

```python
# opendm/model/dm05/dm05_arch.py:907-928（有删节）
v_t = self._action_output_proj(suffix_out)
elem_mse = F.mse_loss(v_t, u_t, reduction="none")     # [B, T, D]
per_sample_fm = (elem_mse * action_mask).sum(dim=(1, 2)) / action_mask.sum(dim=(1, 2))
fm_loss = per_sample_fm.mean()
loss = fm_loss
...
return DM05OutputWithPast(loss=loss, fm_loss=fm_loss, logits=None, past_key_values=None)
```

全仓检索 `CrossEntropy` / `lm_loss` / `language_loss` / `text_loss` / `labels=`：**只命中 `logits=None` 这一处**，没有任何 CE / LM loss。

**② 无自回归解码路径**
模型只有 `forward()`（flow matching）与 `inference_action()`（10 步 Euler 积分），不存在文本生成意义上的 `generate()`。

**③ prompt 中没有 CoT 结构**
`ChatTokenization`（`transforms.py:716-781`）只拼：

```text
Robot: {robot_type}
Control mode: {control_mode}
Overall speed: {speed}
Task: {prompt}.
History images: {<unused1>/<unused0> 槽位}
{prompt} image: <image> ...
States: {256-bin 离散化状态}
```

**没有任何"当前阶段 / 下一步 / 未来事件 / 动作意图"的问题模板**。

**④ 数据 schema 里没有对应字段**
`docs/en/data.md:80-88` 字段表仅含 `images_* / state / prompt / action / is_robot`，没有 subtask / phase / plan。
`docs/en/data.md:94` 明确写道：

> Dexdata dialogue fields such as `answer` and `conversations` are **not required**.

👆 这句话是关键旁证：它暗示**内部 Dexdata 流水线原本是有对话/答案字段的**（很可能就是 CoT 监督的载体），开源版直接把这条线砍掉了。

**⑤ 全历史检索**

```bash
for r in $(git rev-list --all); do
  git grep -l -iE 'chain.of.thought|dynamic programming|monotonic|embodiment cot' $r -- '*.py'
done
# → 命中仅 opendm/exp/dm05_exp.py 中的 time.monotonic() 计时调用
```

40 个 commit（含 `dev/Alex`、`backup/*`、`main`、`origin/main`）**均无**相关实现。同法检索 `dtw / hungarian / linear_sum_assignment / scipy / outlier / static frame / dedup / relabel` 亦零命中。

### 2.2 判定

11 个自回归任务属于**训练侧数据构造 + 额外语言监督头**的组合，开源包里**连数据 schema 都没保留**。

---

## 3. Trajectory Alignment Layer —— ❌ 完全未实现

### 3.1 现状：教科书式的"固定时间索引对齐"

正是博客要解决的那个问题：

```python
# opendm/data/transforms.py:311-335  BuildActionChunk.__call__
if "action" in data:
    read_key = "action"; start = frame_index
else:
    read_key = "state";  start = frame_index + 1

values = []
last_value = None
for step in range(self.action_horizon):
    raw_idx = start + step                      # ← 固定索引步进
    if raw_idx <= episode_term:
        frame = orjson.loads(lines[raw_idx])
        last_value = np.asarray(frame[read_key], dtype=np.float32)
    values.append(last_value)                   # ← 越界则重复最后一帧

data["action"] = np.stack(values, axis=0)[None, ...]
data["action_mask"] = np.ones_like(data["action"], dtype=bool)   # ← 恒为全 1
```

### 3.2 逐条对照博客机制

| 博客机制 | 代码现状 |
| --- | --- |
| 模型输出定长未来动作段（50 步） | ✅ `chunk_size=50`（`dm05_exp.py:76`） |
| 数据侧保留更细粒度真值轨迹 | ❌ 只按 `action_horizon=50` 切，没有"更细真值"概念 |
| 每个预测动作匹配一个**动作锚点** | ❌ 一一对应固定偏移 `start + step` |
| 匹配**严格单调** | ❌ 结构上恒单调，但无约束、无求解 |
| **动态规划**求解最小总匹配损失 | ❌ 无（`scipy` / `linear_sum_assignment` / `dtw` 全仓零命中） |
| 相邻锚点**轨迹连续性**防"挑软柿子" | ❌ 无 |

### 3.3 两点避免误读的补充

- `action_mask` 恒为全 1，意味着**越界填充的帧也参与 loss**。对比 starvla PI0 会用 `action_pad` 屏蔽尾部 pad timestep（`PI0.py:811-815`），开源 OpenDM 在这一环反而更粗。
- 唯一与"进度"沾边的是 **speed 文本条件**（`transforms.py:729-733`，`"Overall speed: {speed}"`）。它把快慢作为**语言条件**喂给模型，而非在 loss 侧做进度对齐——可视为博客思路的**廉价近似**，但不是 Trajectory Alignment Layer。

---

## 4. 数据清洗流水线 —— ❌ 基本未实现

### 4.1 开源侧数据路径的全貌

```text
JsonlDataset → LoadImages / LoadHistory → PixelTransform → Normalize
             → ChatTokenization → PadAction
```

（`opendm/exp/dm05_exp.py:346-378` 的 `build_dataset`）

### 4.2 五条策略逐条对照

| 策略 | 现状 | 唯一沾边的东西 |
| --- | --- | --- |
| 离群剔除 | ❌ | `Normalize` 在做归一化时 `np.clip(arr, lo, hi)`（`transforms.py:135-137`）——**数值裁剪，不是样本剔除**；ROS 异常值、物理可达性、视觉-状态-动作一致性检查均无 |
| 静态帧剔除 | ❌ | 全仓 `stationary / is_static / motion_thresh` 零命中 |
| 低价值动作剔除 | ❌ | 无；`is_robot` 字段注释明确"当前 DM05 训练流程不会用它构造动作目标"（`docs/en/data.md:85`） |
| 动作模式去重 | ❌ | 无 |
| 错误标注重打 | ❌ | 无（也没有 `subtask` 标签可打） |

### 4.3 现有的两处"清洗"（均为工程性）

1. `opendm/data/dataset.py:73-79`：`_load_jsonl` 过滤空行；
2. `opendm/data/dataset.py:33-39`：`range(max(0, num_samples - 1))` —— 每个 episode 丢弃最后一帧（末帧没有未来动作可切）。

### 4.4 其他

`third_party/` 仅含 `robochallenge_inference` 与 `vla_arena` 两个**评测客户端**，不含训练侧清洗代码。

---

## 5. 与 `starvla/starVLA/model/framework/VLM4A/PI05.py` 的对比

### 5.0 前置说明

两者**不是同一层次的东西**：

- `PI05.py` 是 StarVLA（通用 VLA 训练框架）里的一个 framework 插件；
- opendm 是 DM0.5 的**官方全栈开源实现**（模型 + 数据 + 训练 + 服务 + TRT/CUDA Graph 加速，约 11k 行 Python）。

### 5.1 代码形态

| 项目 | opendm DM05 | starvla PI05.py |
| --- | --- | --- |
| 体量 | `model/dm05/dm05_arch.py` 1596 行 + `data/transforms.py` 922 行 + `infer/` 4000+ 行 | **133 行**（含大段中文注释）；**99% 逻辑继承 `PI0.py`（1095 行）**，只覆盖 3 个类属性 |
| 扩展方式 | 自研 `DM05ForConditionalGeneration` / `DM05ActionExpert` | `default_config_cls` / `action_head_cls` / `model_name` / `discrete_state_input` 四个类属性（`PI05.py:130-133`） |

### 5.2 架构对比（同源 π0.5，规模与工程取舍不同）

| 维度 | opendm DM05 | starvla PI05 |
| --- | --- | --- |
| VLM 主干 | **Gemma 3 4B**（hidden 2560、34 层、SigLIP-so400m、256 token/图） | **PaliGemma 3B**（width 2048、depth 18、SigLIP-400m、224×224，`PI05.py:73,79-88`） |
| Action Expert | 34 层 / width 1024 / 8 heads / head_dim 256（≈680M） | 18 层 / width 1024 / 8 heads / head_dim 256（≈300M，`PI05.py:91-100`） |
| 时间条件 | `adarms_cond`：`posemb_sincos(max_period=4.0)` → `time_mlp_in` → SiLU → `time_mlp_out` → SiLU；逐层 3×hidden 调制 + `final_time_modulator` | `OpenPI05ActionHead.use_adarms=True`（`OpenPI_ActionHead.py:283-305`），**结构同构**：`F.silu(time_mlp_out(F.silu(time_mlp_in(time_emb))))`，`max_period=4.0` |
| state 输入 | 归一化 → 256 bin 离散化 → 写入 prompt `States: ...`（`transforms.py:768-781`） | PI05 `discrete_state_input=True` → prompt `"Task: ..., State: ...;"`；**suffix 中无 `state_proj`**，与 DM05 一致 |
| 前缀/后缀注意力 | prefix 单次前向产 **KV cache**，suffix 单向 attend prefix（suffix 内部双向、非 causal）——π0.5 部署形态（`dm05_utils.py:275-315`） | 双流**逐层共享 self-attention**（q/k/v 拼接 + `att_masks` 块状掩码）——π0 原始形态（`PI0.py:661` `forward_shared_gemma_layer`） |
| chunk / action_dim / 采样步 | 50 / 32 / 10 步 Euler | 50 / 32 / 10 步 Euler（**完全一致**） |
| 训练 loss | 纯 flow matching，`Beta(1.5, 1.0)` 时间采样，`loss = fm_loss`（`dm05_arch.py:858-917`） | 纯 flow matching `MSE(u_t, v_t)`（`PI0.py:807`） |
| **历史上下文** | ✅ 见第 1 节 | ❌ 无（全仓 `history` 仅命中 GRU state-history 与 action-history 死代码） |
| **CoT 监督** | ❌ | ❌（同目录另有 `LangForce.py` 做 language-forcing / LLR 语言监督，但**不是** 11 任务 CoT） |
| **轨迹对齐** | ❌ | ❌ 索引式切片 + `first_last`/`zero` padding（`gr00t_lerobot/datasets.py:386`）；但**有 `action_pad` 尾部 mask** 让 pad 步不参与 loss |
| 推理加速 | 手写 big-kernel、**TensorRT vision engine**、**CUDA Graph suffix replay**、flex_attention、Liger kernel | `torch.compile` 热路径（`PI0.py:357-392`），无 TRT / CUDA Graph |
| 优化器 | HF Trainer + FSDP + **Muon/AdamW 分组**（Muon 专挑 `action_expert.*` 的 2D 矩阵，`muon_adamw.py:41-65`）+ 完整 LoRA 链路 | StarVLA 自带 trainer |

### 5.3 对比结论

- **DM05 = π0.5 架构（adaRMS + 离散 state + 定长 chunk）的"重工程 + 加记忆"版本**：主干更宽更深（Gemma3 4B + 680M expert vs PaliGemma 3B + 300M expert），prefix 改为 KV-cache 形态以便上 TRT / CUDA Graph，并**额外增加了 starvla PI05 完全没有的历史上下文通道**（32 槽 × 16 token = 512 个历史软 token）。
- **PI05.py = π0.5 的"极简复刻"**：代码量约 1/100，架构等价，工程优化依赖 `torch.compile`，**既无记忆、也无 CoT / 轨迹对齐**。
- 因此：**从 PI05 出发能看到的差异，恰好就是 opendm 多出来的 "Context Abstraction Layer" 这一项**；另两项架构级设计（CoT、轨迹对齐）**两家都没有**——差别只是 starvla 在别的 framework（LangForce）里探索语言监督，而 opendm 把数据侧完全留白。

---

## 6. 若要补齐：最小改动切入点

| 目标 | 切入点 |
| --- | --- |
| 历史采样补齐 | `LoadHistory`（`transforms.py:229`）增加 `jitter` / `random_slots` / `history_dropout` 参数；`PixelTransform` 为历史单独拆一条增强 pipeline |
| 历史窗口对齐博客 | 把 MEM playground 的 `max_history_images` 由 32 调到 60，或把 `uniform_fps` 降到 0.5 |
| 训练/推理槽位一致化 | 统一 `dm05_exp.py:290`（`is_history`）、`dm05_exp.py:510`（默认 5）与 `transforms.py:682`（默认 32）三处默认值 |
| CoT 监督 | 数据侧增加 `cot` / `answer` 字段并在 `ChatTokenization` 后拼接 assistant 段；模型侧在 `DM05ForConditionalGeneration.forward`（`dm05_arch.py:826`）增加一路 CE loss，与 `fm_loss` 加权求和 |
| 轨迹对齐 | 用锚点匹配版替换 `BuildActionChunk`（`transforms.py:271`）：保留细粒度真值 + 严格单调 DP 求解；同时把 `action_mask`（`transforms.py:335`）由全 1 改为由匹配结果 / 尾部 pad 决定 |
| 数据清洗 | 新增 transform 插入 `Pipeline` 最前（`dm05_exp.py:346`）；给 `JsonlDataset._build_index`（`dataset.py:26`）加样本级过滤钩子 |

---

## 7. 核查方法说明（可复现）

```bash
# 1) 关键字全仓检索（排除 third_party / __pycache__）
grep -rn --include='*.py' -iE 'cot|chain.of.thought|reasoning|trajectory_align|dynamic.?programming|monotonic|outlier|static.?frame|dedup|relabel' .

# 2) 全历史检索（40 个 commit，含所有分支）
for r in $(git rev-list --all); do
  git grep -n -iE 'chain.of.thought|dynamic programming|monotonic|embodiment cot' $r -- '*.py'
done
# → 仅命中 opendm/exp/dm05_exp.py 的 time.monotonic()

# 3) 符号全量清点，确认 data/ 与 model/ 下无遗漏模块
grep -rn -E '^class |^def ' opendm/data/*.py opendm/dataset/*.py opendm/model/dm05/*.py

# 4) 确认无语言监督损失
grep -rn --include='*.py' -E 'CrossEntropy|lm_loss|language_loss|text_loss|labels=' opendm/
```

---

## 8. 附：核查过的关键文件清单

| 文件 | 与本报告的关系 |
| --- | --- |
| `opendm/data/transforms.py`（922 行） | 功能 1 数据侧、功能 3、功能 4 的主战场 |
| `opendm/data/collator.py`（165 行） | 功能 1 的 batch 打包 |
| `opendm/data/dataset.py`（103 行） | 功能 4 仅有的两处"清洗" |
| `opendm/data/normalize.py`（397 行） | RunningStats / 分位数统计 |
| `opendm/data/augmentations.py`（106 行） | 图像增强（当前帧与历史共用） |
| `opendm/model/dm05/dm05_arch.py`（1596 行） | 功能 1 模型侧 + 功能 2 缺失的直接证据（`forward` 只有 FM loss） |
| `opendm/model/dm05/dm05_utils.py`（315 行） | `<unused1>` pad 屏蔽、suffix attention mask |
| `opendm/infer/dm05_trt_utils.py`、`dm05_infer.py` | 功能 1 推理侧 / TRT 快路径 |
| `opendm/exp/dm05_exp.py`（1266 行） | 数据/推理配置、学习率分组、服务入口 |
| `opendm/optimizer/muon_adamw.py`（535 行） | 分离学习率组的实现能力 |
| `docs/en/data.md`、`docs/en/dm05_mem_sft.md` | 数据 schema 与 MEM 历史的官方口径 |
| `third_party/`（robochallenge_inference / vla_arena） | 仅评测客户端，无训练侧功能 |
