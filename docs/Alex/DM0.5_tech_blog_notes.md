# DM0.5 技术博客整理：为开放世界而设计，泛化自然涌现

- 整理时间：2026-09-21 11:22 (CST)
- 原文标题：*DM0.5: Designed for the Open World, Where Generalization Emerges.*
- 原文地址：<https://www.dexmal.com/blog/dm0.5/index_en.html>
- 发布方：Dexmal（原力灵机）
- 文档性质：对官方技术博客的**结构化整理 + 方法要点提炼**。第 1–10 节忠实于原文；**第 11 节的代码对应关系是整理者自行核对本仓库得出的，不属于原文内容**。

> **关注重点说明**：按要求，本文档把**模型方法（Model）**作为核心，第 3–6 节为方法主线；Data / Experiments 部分做了压缩。

---

## 1. 一句话概括

DM0.5 是 Dexmal 的第二代**具身原生基础模型**（embodied-native foundation model），基于 VLA 架构，目标是从"实验室里跑固定任务"走向"**开放世界里的零样本泛化**"。

相对 2026 年 2 月发布的第一代 DM0，DM0.5 的关键不是"参数更大、数据更多"，而是四方面的系统性改造：**历史上下文建模、具身推理监督、动作监督对齐、数据质量**——把模型从"由当前帧驱动的策略"推向"能理解任务进度、处理开放式指令、稳定输出连续动作的具身基础模型"。

**五个核心能力（原文自述）**

| 能力 | 说明 |
| --- | --- |
| Zero Shot | 把零样本能力扩展到**未见过的开放环境**，按自然语言指令完成操作任务 |
| Fine-Tuning | 更强的基座 → 更强的下游专家策略；微调所需**数据量与算力更少**、质量更高 |
| Long-Horizon Memory | 架构支持历史观测，可纳入**最长 60 秒**的任务历史 |
| Robust Actions | 光照、相机视角变化、人为主动干扰下策略行为**稳定** |
| Multi-Embodiment | 多机器人多任务训练，可通过后训练迁移到**预训练时未见过**的本体 |

---

## 2. 模型总体架构

- **4B 视觉语言模型（VLM）作为主干（backbone）**
- **680M Action Expert** 负责生成连续机器人动作

与 DM0 的差异定位：不止是更大、更多数据，而是围绕**长程依赖、语义推理、数据噪声、动作连续性**四个真实机器人任务中的痛点做的设计。

> 原文配图：*DM0.5 model architecture*

---

## 3. 核心方法速览（重点）

DM0.5 的三项架构级设计 + 一条数据侧流水线，分别对应一类具体问题：

| # | 方法组件 | 解决的问题 | 机制关键字 |
| --- | --- | --- | --- |
| 1 | **Context Abstraction Layer**（上下文抽象层） | 传统 VLA 只看当前帧，短程/近似马尔可夫任务够用，长程任务丢状态 | 历史帧 slot、时间+空间采样、压缩为定长 visual token、随机历史长度与历史增强 |
| 2 | **Embodiment CoT Tasks**（具身思维链任务） | 只有连续动作监督，模型"知其然不知其所以然" | 在机器人数据上引入 **11 个自回归任务**，三类：任务规划 / 事件与环境预测 / 动作生成 |
| 3 | **Trajectory Alignment Layer**（轨迹对齐层） | 遥操作数据节奏不一，按固定时间戳对齐会让模型学到"采集节奏"或过拟合时间噪声 | **动态动作匹配**：按**轨迹进度**而非时间索引对齐，严格单调 + 动态规划求解 |
| 4 | **数据清洗流水线** | 多源多本体多任务数据噪声大 | 离群剔除、静态帧剔除、低价值动作剔除、动作模式去重、错误标注重打 |

一句话串起来：**看更长的历史（1）→ 想清楚为什么（2）→ 动作监督对齐到任务进度而非时间（3）→ 数据本身先干净（4）**。

---

## 4. 方法详解一：Context Abstraction Layer（历史上下文融合）

**动机**：传统 VLA 在每个控制步只接收当前图像 + 当前机器人状态。这对短程、局部、近似马尔可夫的任务足够，但长程任务中很多关键条件**并不在当前帧里**。

**推理时**：模型接收当前帧 + 过去若干关键帧，从而获得**最长约 1 分钟**窗口内的任务状态变化信息，例如：

- 物体是在哪里被拿起的
- 某个工具是否已经用过
- 某个区域是否已经清理过
- 机器人是否已经经过某个地标

**训练时**：

- 数据流水线在当前时刻之前采样多个**历史 slot**；
- 每个历史 slot 依次经过**时间采样（temporal sampling）**与**空间采样（spatial sampling）**，压缩为**固定数量的 visual token**；
- 使用**随机历史长度（randomized history lengths）**与**历史增强（history augmentation）**，让模型同时学会处理长历史、短历史、以及完全没有可用历史的情况。

**效果**：降低对固定历史窗口的依赖；当历史缺失或部分损坏时，能够**优雅退化为当前观测驱动的行为**（fall back gracefully）。

---

## 5. 方法详解二：Embodiment CoT Tasks（具身推理监督）

**动机**：具身模型需要建立关于**环境变化、指令语义、自身本体状态与行为**的强表征，而**语言是这类学习最高效的表征接口之一**。

**做法**：在机器人数据上引入 **11 个自回归任务**，使训练不再仅由连续动作监督驱动，同时强化指令跟随、动作预测与时间维度的场景理解。给定当前图像、机器人状态、任务指令与历史上下文，模型需回答**受约束的问题**。

**三类具身推理任务**

| 类别 | 关注点 | 作用 |
| --- | --- | --- |
| **Task Planning**（任务规划） | 当前任务阶段、前后步骤关系、整体进度 | 理解"已经做了什么、接下来该做什么" |
| **Event & Environment Prediction**（事件与环境预测） | 任务边界、状态转移、关键未来事件 | 强化对场景演化与阶段切换的感知 |
| **Action Generation**（动作生成） | 未来动作 / 动作意图的语义摘要 | 在生成连续动作前形成更清晰的表征 |

**本质变化**：机器人数据从"只有动作监督"扩展为**指令理解 + 时间推理 + 动作生成的联合监督**。模型学到的不只是"当前图像对应哪个动作"，还包括"**在给定指令与任务进度下，为什么这个动作是合适的**"以及"世界状态之后会怎么变"。这提升了复杂长程任务中的指令跟随、动作连贯性与任务完成度。

---

## 6. 方法详解三：Trajectory Alignment Layer（动态动作匹配）

**动机**：遥操作采集的机器人数据在执行节奏上差异很大——同一个任务在不同演示中的完成速度不同。如果强迫模型把预测动作对齐到原始轨迹的**固定时间戳**，模型可能学到**数据采集的节奏**，或过拟合到**时间噪声**，而不是任务相关的动作结构。

**机制：按轨迹进度对齐，而非按时间索引对齐**

1. 模型输出**定长**的未来动作段（fixed-length segment），数据侧保留更细粒度的真值动作轨迹；
2. 训练时，每个预测动作匹配到真值轨迹中的一个**动作锚点（action anchor）**；
3. 匹配必须**严格单调（strictly monotonic）**：靠后的预测动作只能匹配轨迹中更靠后的位置，**防止时间倒转**；
4. 每个候选匹配的 loss 由"预测动作 vs 对应真值锚点"的误差计算；
5. 整体匹配通过**动态规划（dynamic programming）**求解，最小化所有预测动作的总匹配损失。

**防"挑软柿子"**：为避免只选出局部容易的动作，DM0.5 还考虑相邻锚点之间的**轨迹连续性**——光靠单个动作点接近不够，相邻锚点之间的真值轨迹也应能被预测动作的变化合理"解释"。这样既允许执行速度上的合理变化，又降低模型**跳过任务关键阶段**的风险。

**效果**：降低遥操作数据中的**时间相位噪声**；引导模型关注抓取、对齐、接触、释放等**任务关键变化**；学到的**动作速度场（action velocity field）**更平滑、更鲁棒，从而提升在不同演示者、不同执行节奏之间的泛化。

---

## 7. 训练策略（Training）

**核心思路：数据对齐之后，做多源混合训练。**

| 数据类型 | 提供什么监督 |
| --- | --- |
| 机器人操作数据 | 动作学习的**主要监督** |
| 视觉-语言数据 | 保持视觉主干的**开放词表理解与空间推理**能力 |
| 导航数据 | 长指令理解与路径决策监督 |
| 视频理解数据 | 时间建模、事件理解、动态场景表征 |

**优化细节**

- VLM 主干与 Action Expert 使用**分离的学习率组（separate learning-rate groups）**；
- VLM 主干用**更小的学习率**，以减少**灾难性遗忘**、保留通用视觉-语言能力；
- Action Expert 用**更大的学习率**，更有效地学习机器人动作分布；
- 训练采用**混合精度 + 分布式优化**；
- 针对长上下文、多相机输入与历史 token 带来的算力开销，对 Action Expert 做了**额外优化**。

---

## 8. 推理（Inference）

- 以 **action chunk** 方式推理；
- 默认使用 **10 步 diffusion / Flow Matching** 生成 **50 步动作块**；
- 优化后吞吐：**单张 NVIDIA RTX 4090 上 10 Hz**，**单张 NVIDIA H100 上 20 Hz**。

---

## 9. 数据（Data）

### 9.1 数据构成

大规模异构数据预训练，覆盖机器人操作、具身导航、第一人称人类操作、通用多模态视觉-语言数据。

| 类别 | 内容 |
| --- | --- |
| **机器人操作数据** | 多本体的真实操作数据：AgileX ALOHA、Galaxea R1 Lite、AgiBot G1、Franka Emika Panda、UR5、ARX5，以及 Dexmal 自研的双臂移动操作机器人 |
| **具身导航数据** | 开源视觉-语言与开放词表导航数据集，以及基于 **3D 重建场景**自采的导航数据 |
| **第一人称人类操作数据** | 日常生产环境中采集的第一人称操作，覆盖手-物交互、工具使用、物体操作、细粒度原子动作 |
| **通用多模态视觉-语言数据** | 图像、视频、视觉指令数据，并用自动生成流水线增强：空间 grounding、未来状态预测、动作结果分析、**反事实推理（counterfactual reasoning）** |

### 9.2 数据清洗策略（五条）

| 策略 | 具体做法 |
| --- | --- |
| **离群剔除** | ROS 日志中机器人偶发异常大/异常小/不连续的值；过滤明显超出物理可达范围或违反运动连续性的记录；检查视觉输入、机器人状态、动作标签三者一致性，剔除"画面有明显运动/抖动但状态与动作记录未反映该运动"的片段 |
| **静态帧剔除** | 图像与机器人状态长时间静止的片段动作信息量低、拖低训练效率、且可能损害部署时的动作响应性，训练前移除 |
| **低价值动作剔除** | 执行不完整、意图不清、与当前任务目标无关的行为会引入噪声监督，降低策略学习的稳定性与泛化性，予以移除 |
| **动作模式去重** | 某些平台（如 ALOHA）不同关节组合可能对应几乎等价的末端运动模式；去重以避免模型学到不一致、互相混淆的关节映射，保留统一一致的关节表示 |
| **错误标注重打** | 原始数据集可能含错误任务标注；构建**自动重打标流水线**，通过**跨模态一致性检查**校验并修正子任务标签，提升标签可靠性与数据利用率 |

---

## 10. 实验（Experiments）

### 10.1 零样本泛化

**评测设置**：沿两个维度系统测量——**动作类型**与**条件约束**。

- 动作维度：8 个基本操作原语 —— pick、put、move、pull、cover、wipe、stack、press
- 条件维度：7 类语义约束 —— color、shape、size、status、sequence、relative position、absolute position
- 四组"模型 × 平台"配置：Franka 平台上的 **Pi0.5-Droid** 与 **DM0.5-Droid**；Dexmal-Mirror 平台上的 **DM0** 与 **DM0.5**

**结果**：DM0.5 在多数评测维度上优于 Pi0.5-Droid 与 DM0，体现更强的指令理解与执行能力；整体呈现出更系统的零样本动作执行与语言条件操作能力——动作覆盖更广、基本操作更稳定、多类语义约束下的指令跟随更强。

> 原文配图：*DM0.5 zero-shot success rates on Franka and Dexmal-Mirror*（Table 1 / Table 2，原文网页以图表呈现，正文未给出完整数值）

### 10.2 微调能力

**真实机器人操作：RoboChallenge Table30 v2**

- 覆盖长程记忆、多步顺序执行、视觉感知与目标定位、精细抓放、工具交互、双臂协调等真实桌面场景；
- **整体成功率 43%，综合 Score 54.42，达到 SOTA**；
- 需要记忆目标状态与动作顺序的任务（如盖章定位、按钮按压）中，**历史上下文建模**提升了执行稳定性；
- 精细操作（如插花，需要物体识别 + 抓取位姿选择 + 精确放置）体现强视觉 grounding 与末端控制；
- 双臂任务（如双手端托盘）能保持相对位姿与搬运稳定性。

**仿真操作基准**

- 单臂 **LIBERO** 与双臂 **RoboTwin2.0**：微调后适配性强，在对比方法中达到 SOTA 水平。

**仿真导航基准**

- **R2R Val-Unseen**：DM0.5-Nav 在 Navigation Error、Oracle Success、Success Rate 上均为最佳；
- **RxR Val-Unseen**（更具挑战）：DM0.5-Nav 在列出的全部四项指标上排名第一。

### 10.3 历史上下文建模

**为什么重要**：具身任务中很多关键条件并不总在当前帧可见——物体可能已被移走、桌面状态可能已被先前动作改变、任务规则可能只在 episode 开头由人类演示出现过。

**两个真机实验（Dexmal-Mirror）**

| 实验 | 记忆尺度 | 内容与结论 |
| --- | --- | --- |
| **"拿起杯子并擦桌子"** | 短程记忆 | 机器人须先拿起杯子露出下方区域，再擦桌，最后把杯子放回**原位**；杯子拿起后初始位置在当前帧已不可见，DM0.5 依靠**历史视觉记忆**恢复其初始位置并在末尾复原 |
| **"从人类演示中学习"** | 长程记忆 | 任务开头人类先演示一条规则；机器人须观察演示中电池的摆放方式，并在后续自身执行阶段**保持该规则一致**；DM0.5 利用**长历史视觉上下文**把早期演示转化为后续操作策略 |

**结论**：两者展示了互补时间尺度上的上下文记忆能力；说明预训练阶段形成的**上下文抽象**可以在下游监督微调（SFT）中被激活，并在视觉-语言条件下参与动作预测。

### 10.4 策略鲁棒性

**相机视角**（Franka 平台）

- 部署配置：**1 个腕部相机**（固定在末端执行器）+ **2 个可独立移动的第三人称相机**；
- 评测 **9 种相机配置**（左侧 3 种位姿 × 右侧 3 种位姿），每种配置下连续执行 **10 次**抓放试验：6 次标准桌面高度 + 4 次放置目标抬高；
- 结果：尽管第三人称相机位置变化，模型在各配置下**成功率保持稳定偏**高；
- **两阶段策略**（轨迹分析发现）：第一阶段用第三人称相机的**全局场景信息**把末端引导到目标物体附近；第二阶段主要依赖固定腕部相机的**局部视觉反馈**做精细调整与对齐；
- 当某一第三人称相机处在极端视角（如 Left3 / Right3）时，第一阶段粗定位的空间偏移更明显，但多数试验中模型仍能靠腕部相机的局部修正**补偿偏移**并完成任务。

**人为扰动**（Dexmal-Mirror）

- 目标被人类移动或**临时遮挡**时，DM0.5 仍能保持场景理解并继续执行任务；
- 容器位置与朝向变化时，模型**不会盲从原固定轨迹**，而是按更新后的视觉状态调整末端位置与动作方向，继续操作目标；
- 说明对动态场景的强适应性与强任务连续性：当外部扰动改变"目标物体—容器—机械臂"的空间关系时，模型能**重建三者的对应关系**，避免任务中断或动作失败。

> 原文配图：*Franka camera positions and test success rates*、*Continued execution under camera perturbation*、*Dynamic adaptation after the target object and container are moved by a human*

---

## 11. 附：与本仓库 OpenDM 实现的对应关系（整理者核对，非原文内容）

本仓库 `/root/workspace/opendm` 即 DM0.5 的开源实现（权重 `checkpoints/DM05`，架构 `DM05ForConditionalGeneration`）。核对后情况如下：

**（1）架构规格——与原文完全吻合**

| 项目 | 原文/权重配置 | 核对结果 |
| --- | --- | --- |
| VLM 主干 | 4B | Gemma 3 4B：`hidden_size=2560`、34 层、`intermediate_size=10240`、vocab 262208 |
| 视觉塔 | — | SigLIP-so400m：`hidden_size=1152`、27 层、patch 14；`mm_tokens_per_image=256` |
| Action Expert | 680M | `action_config`：`hidden_size=1024`、34 层、8 heads、`head_dim=256` |
| 动作块 | 50 步 | `chunk_size=50`、`action_dim=32` |

**（2）三项核心方法在开源代码中的落地情况**

| 博客方法 | 开源实现 | 证据 |
| --- | --- | --- |
| Context Abstraction Layer（历史上下文融合） | ✅ **已实现** | `opendm/model/dm05/dm05_arch.py`（历史 slot 经 `<unused0>` 注入、`<unused1>` 为无效历史占位并在注意力中屏蔽）、`opendm/data/transforms.py`（`is_history` / `max_history_images`、历史图像 PadToSquare+Resize）、`opendm/data/collator.py`（`history_mask` / `history_pixel_values` 打包）、`opendm/infer/dm05_infer.py` 与 `dm05_trt_utils.py`（TRT 下 current + 补零历史拼批、历史特征池化到 16 token） |
| Flow Matching 动作生成 | ✅ **已实现** | `dm05_arch.py` 中 `forward()` 为 flow matching 训练，suffix 前向 + `action_expert` |
| Embodiment CoT（11 个自回归任务） | ⚠️ **未在开源代码中找到** | 全仓（排除 `third_party`）检索 `CoT` / `reasoning` / `planning` 等，仅命中 tokenizer 词表，无对应任务实现或数据构造 |
| Trajectory Alignment Layer（动态动作匹配 / 动态规划对齐） | ⚠️ **未在开源代码中找到** | 检索 `monotonic` / `matching` / `trajectory_align` 等，仅命中 `time.monotonic()` 计时调用 |
| 数据清洗流水线（五条策略） | ⚠️ 未见完整实现 | 开源侧以数据集/transform/collator 为主，未发现离群剔除、静态帧剔除、低价值动作剔除、动作模式去重、自动重打标的完整流水线 |

**（3）其他已核对到的实现细节**

- **分离的学习率组（机制存在，取值需注意）**：`opendm/optimizer/muon_adamw.py` 的 `is_default_muon_parameter()` 专门挑选 `model.action_expert.*` 下的 2D 层矩阵（排除 embedding / norm）交给 **Muon** 优化器，其余参数走 AdamW；`muon_lr_scale` 是给这一组单独缩放学习率的旋钮。但该旋钮**默认值为 1.0**（`base_lr` 默认 2.5e-5），代码里看不到"Action Expert 用更大学习率"的实际取值——原文的"VLM 小 lr / Action Expert 大 lr"属于训练配方的选择，开源代码只提供了**实现该配方的能力**；
- **推理加速**：`opendm/infer/dm05_infer_arch.py` 为 Action Expert 提供了 big-kernel 前缀/后缀、CUDA Graph、TRT 等快路径（对应原文"对 Action Expert 做额外优化以支撑长上下文、多相机、历史 token 的开销"）；
- **历史推理的服务形态**：`opendm/exp/dm05_exp.py` 中服务需以 `--data-config.is-history` 启动才接受 `history_images`，单次请求历史图像数上限由 `max_history_images`（默认 5，dataset 侧默认 32）控制。

> **一句话结论**：开源版本覆盖了 **架构 + 历史上下文融合 + Flow Matching 训练/推理加速**；而 **Embodiment CoT 监督**与**动态动作匹配**这两项训练侧关键设计**未随代码开源**，只能从博客描述理解。

---

## 12. 原文配图清单（网页中的图，正文未给数值）

1. DM0.5 open-world manipulation capabilities
2. 开放环境中的餐桌布置（Table arrangement in an open environment）
3. 跨容器空间约束下的物体放置（Object placement under cross-container spatial constraints）
4. 执行多目标、带约束的指令（Executing multi-target, constrained instructions）
5. 多步堆叠与精确对齐（Multi-step stacking and precise alignment）
6. DM0.5 model architecture
7. DM0.5 zero-shot success rates on Franka and Dexmal-Mirror
8. DM0.5 results on LIBERO and RoboTwin2.0 simulation benchmarks
9. DM0.5 results on R2R and RxR navigation benchmarks
10. Short-horizon memory: pick up the cup, wipe the table, and restore the cup
11. Long-horizon memory: follow an early human demonstration to place the battery
12. Demonstration learning with rule consistency across execution stages
13. Franka camera positions and test success rates
14. Continued execution under camera perturbation
15. Dynamic adaptation after the target object and container are moved by a human

---

## 13. 结论（原文）

> 我们相信，为开放世界而构建，不是往模型里塞进越来越多的任务；而是拒绝"一张任务清单就能代表世界"的幻觉。
>
> 通往通用具身智能的更快的路，是走出脚本化的环境，用尽可能多地暴露于真实世界复杂性的方式来训练模型。
>
> DM0.5 是这条路上的一步。它给机器人更长的记忆、更开放的理解、更稳定的行为。更重要的是，它让机器人系统更接近在**没有预设答案的世界**中运行。
>
> 今天我们仍需采集机器人数据、设计任务、构建评测。未来，我们希望**真实世界本身成为最好的老师**——每一次交互、每一次尝试、每一次成功与失败，都能成为机器人理解世界并自我改进的一部分。最终，机器人应当不止于复现见过的动作，而是用对世界的理解去完成**从未被写进脚本**的任务。
>
> DM0.5 只是这个故事的开端。
