# DM05 数据 → `input_ids` 全链路详解（小白版）

- 记录时间：2026-10（与《DM05模型结构与训练推理详解_小白版.md》同批）
- 面向读者：会一点 Python、但**没读过数据管线代码**的人。不需要懂 Transformer。
- 全文只回答一个问题：**磁盘上的一行 JSONL，是怎么变成模型手里的 `input_ids [B, 1536]` 的？**
- 代码基线：`opendm/data/dataset.py`、`opendm/data/transforms.py`、`opendm/data/collator.py`、`opendm/dataset/*.py`、`playground/dm05_mem_sft_robodojo_cover_blocks.py`
- 实测来源：权重目录 `/mnt/cfs/opendm/checkpoints/DM05-MEM-Robodojo-Sim`（tokenizer / processor / chat template）+ 原文档 3.2 节的真机样本
- 配套阅读：同目录 `DM05模型结构与训练推理详解_小白版.md` 的 1.6（token 布局）、2.2（数据 → 张量）

---

## 0. 一句话结论 + 全局图

**`input_ids` = "模板文本 + 3 张当前图的占位符 + 历史图占位符 + States 文本"，交给 Gemma3 processor 渲染、分词、右侧补齐 1536 的产物。**

```text
磁盘：data/.../episode_0000001.jsonl 的第 123 行
        │  ① JsonlDataset.__getitem__
        ▼
一个 dict：{prompt, images_1..3, state, action} + meta_data{...}
        │  ② Pipeline 里 5 个"备料" transform（不产生 token）
        │     BuildAction → LoadImages → LoadHistory → PixelTransform → Normalize
        ▼
        │  ③ ChatTokenization：把原料拼成 messages
        ▼
messages = [{"role":"user","content":[text, image, text, image, ...]}]
        │  ④ processor.apply_chat_template(tokenize=True, add_generation_prompt=True)
        ▼
input_ids [1, L]（L ≈ 1213~1243）+ attention_mask + token_type_ids + pixel_values
        │  ⑤ 后处理：history_mask / token_type_ids
        │  ⑥ TrainingCollator：右侧补到 1536 + 沿 batch 维拼
        ▼
input_ids [B, 1536]
```

**最终 batch 里的字段**（RoboDojo 场景，B = batch size）：

| 字段 | 形状 | 一句话 |
| --- | --- | --- |
| `input_ids` | `[B, 1536]` | 本文主角：prompt 的 token id，右侧 pad 到 1536 |
| `attention_mask` | `[B, 1536]` | 1 = 有效，0 = 右侧 padding（`<unused1>` 也算 1，模型内部再屏蔽） |
| `token_type_ids` | `[B, 1536]` | 图像类座位（含历史槽）为 1，文本为 0 |
| `pixel_values` | `[B, 3, 3, 448, 448]` | 3 路当前相机（B=1 时是 `[3,3,448,448]`） |
| `history_pixel_values` | `[ΣHᵢ, 3, 448, 448]` | batch 内所有历史图**沿 0 维拼在一起**（每样本 Hᵢ ≤ 20） |
| `history_mask` | `[B, 1536]` bool | 每行 True 恰好 Hᵢ×16 个，标出 `<unused0>` |
| `action` / `action_mask` | `[B, 50, 32]` | 未来 50 帧动作（14 维真实 + 18 维 pad）及其掩码 |

---

## 1. 四个名词先分清

| 名词 | 是什么 |
| --- | --- |
| **样本（sample）** | JSONL 里的一行 = 某一集的某一帧 |
| **transform** | 一个"接收 dict、加工、再返回 dict"的小函数（如 `LoadImages`） |
| **Pipeline** | 一串 transform 按顺序串起来，`for t in transforms: data = t(data)`（`transforms.py:33-46`） |
| **messages** | HuggingFace chat 格式：`[{"role":"user","content":[...]}]`，是"渲染成 token"之前的**中间态** |

> 关键认知：**transform 阶段全程不碰 token**，它只是往 `data` 这个 dict 里塞 `images` / `history_images`；真正把 dict 变成 `input_ids` 的只有一个环节——`ChatTokenization` 里的 `apply_chat_template`。

---

## 2. 第 0 步：磁盘上的一行长什么样

### 2.1 JSONL 的行结构

RoboDojo 的 LeRobot v3.0 原生格式被 `script/robodojo_lerobot_to_opendm.py` 转成 OpenDM 的 JSONL 布局：**一集 = 一个 `.jsonl` 文件，一帧 = 一行**。每行长这样：

```json
{
  "images_1": {"type": "video", "url": "…/cam_high.mp4",  "frame_idx": 123},
  "images_2": {"type": "video", "url": "…/cam_left.mp4",  "frame_idx": 123},
  "images_3": {"type": "video", "url": "…/cam_right.mp4", "frame_idx": 123},
  "state":  [ …14 个数… ],
  "action": [ …14 个数… ],
  "prompt": "Cover the blocks from left to right, remember their colors, then uncover them in the order: red, green, and blue.",
  "is_robot": true
}
```

注意：**图不是直接存进 JSONL 的**，存的是"哪个视频文件的第几帧"（`url` + `frame_idx`），读图是后面 `LoadImages` / `LoadHistory` 的事。

### 2.2 数据集注册表：`meta_data` 的源头

`dataset_name`（例如 `robodojo_sim_cover_blocks`）会在注册表 `CONVERSATION_DATA` 里查出一条配置（`dm05_exp.py:292-303`）。以 RoboDojo EE 的注册项为例（`opendm/dataset/robodojo_ee.py:23-36`）：

```python
register_dataset({
    "cover": {
        "jsonl_dir":    "./data/robodojo_ee/jsonl",
        "image_dir":    "./data/robodojo_ee/video",
        "image_keys":   ["images_1", "images_2", "images_3"],
        "image_prompts":["Head", "Left wrist", "Right wrist"],
        "robot_type":   RobotType.ALOHA,
        "state_desc":   ROBODOJO_EE_STATE_DESC,
        "fps":          25,
        "speed":        "0.5",
        "control_mode": "eef",
    },
}, prefix="robodojo_ee")
```

这条配置里，除了 `jsonl_dir` / `image_dir` / `image_keys` / `image_prompts` 之外的字段，会被整体塞进每个样本的 `meta_data`（`dm05_exp.py:305-316`）——**这就是 `Robot:` / `Overall speed:` 这些文字的数据来源**。

### 2.3 样本索引：把"很多文件 × 每文件很多帧"压成一维编号

**先说要解决什么问题。** 磁盘上的数据是**两层结构**：

```text
jsonl_dir/
├── episode_0000000.jsonl     第 0 行、第 1 行、第 2 行 …（每行 = 一帧）
├── episode_0000001.jsonl     第 0 行、第 1 行 …
└── …
```

而 PyTorch 的 `Dataset` 只认**一维编号**：`dataset[i]` 要知道"第 i 个样本"是哪一集的哪一帧，`len(dataset)` 要知道总数。所以需要**先建一本"花名册"**，把"（文件, 第几行）"这种二维坐标压成 0、1、2… 的一维编号。

> 类比：图书馆有 200 本书，每本书几百页。你要按"第 i 页"借阅，就得先有一本总目录，写着"第 1 页 → 《A 书》第 1 页；第 2 页 → 《A 书》第 2 页；…；第 400 页 → 《B 书》第 1 页"。

#### `_build_index`（建花名册，`dataset.py:25-39`）

```python
def _build_index(self):
    index_cache = self._get_index_cache(self.jsonl_dir)   # ① 拿到 {文件: 行数}
    file_to_nsamples = index_cache["data"]

    sample_index = []          # ② 花名册：(文件编号, 行号)
    file_to_id = {}            # ③ 文件路径 → 文件编号
    file_id_counter = 0
    for jsonl_file, num_samples in file_to_nsamples.items():
        file_to_id[jsonl_file] = file_id_counter
        for frame_index in range(max(0, num_samples - 1)):      # ④ 注意 -1
            sample_index.append((file_id_counter, frame_index))
        file_id_counter += 1
    self.sample_index = sample_index
    self.id_to_jsonl = {v: k for k, v in file_to_id.items()}    # ⑤ 反查表：编号 → 路径
    self.total_samples = len(self.sample_index)                 # ⑥ = len(dataset)
```

**逐行讲：**

| 行 | 在干什么 | 为什么 |
| --- | --- | --- |
| ① | 读（没有就先生成）`index_cache.json`，内容是 `{"data": {"某文件": 行数, ...}}` | 数一遍所有文件的行数很慢，缓存起来（`dataset.py:81-103`） |
| ② | 准备一个空列表当"花名册" | 每个元素是 `(文件编号, 行号)` 这样一个二元组 |
| ③ | 给每个文件发一个编号：0、1、2… | 花名册里存编号比存一长串路径**省空间、比着也快** |
| ④ | `range(max(0, num_samples - 1))`：**从 0 数到"行数−2"** | 关键！**每集的最后一帧被故意丢掉**（见下面的解释）；`max(0, …)` 只是防御空文件/单行文件的边界情况 |
| ⑤ | 把"路径→编号"反过来变成"编号→路径" | 取样本时手里只有编号，要能查回路径；字典推导式 `{v: k for k, v in …}` 就是"键值互换" |
| ⑥ | 总数 = 花名册长度 | PyTorch 调 `len(dataset)` 时返回它（`dataset.py:62-63`） |

**为什么丢掉最后一帧？**
`BuildAction` 要给当前帧准备"**未来 50 帧**"的动作标签；最后一帧后面没有未来，做不出标签，所以直接不留。`-1` 就是这件事的全部含义。

**举例**：两个文件、分别 5 行和 3 行 →

```text
episode_0000000.jsonl  5 行  →  有效 4 个样本：(0,0) (0,1) (0,2) (0,3)
episode_0000001.jsonl  3 行  →  有效 2 个样本：(1,0) (1,1)

sample_index = [(0,0), (0,1), (0,2), (0,3), (1,0), (1,1)]
                 ↑ idx=0  ↑ idx=1 …            ↑ idx=4  ↑ idx=5
id_to_jsonl  = {0: "episode_0000000.jsonl", 1: "episode_0000001.jsonl"}
total_samples = 6
```

于是 `dataset[4]` 的意思就是："第 1 号文件（B 集）的第 0 行"。

**`index_cache.json`**：第一次运行时由 `build_index_cache` 生成——它会 `glob` 出 `jsonl_dir` 下所有 `*.jsonl`，逐个数行数（`dataset.py:81-103`），之后直接读缓存。所以**这份缓存里只有"行数"，没有"内容"**——内容每次都要真读文件（见 2.4 结尾的提醒）。

### 2.4 `JsonlDataset.__getitem__`：按花名册取一个样本

```python
def __getitem__(self, idx):
    file_index, frame_index = self.sample_index[idx]        # ① 查花名册
    lines = _read_jsonl_lines(self.id_to_jsonl[file_index]) # ② 读整个文件的所有行
    result = orjson.loads(lines[frame_index])               # ③ 只解析"那一行"
    result["raw_lines"] = lines                             # ④ 顺手把整份行列表挂上去
    result["meta_data"] = {                                 # ⑤ 拼 meta_data
        **(self.dataset_meta or {}),                        #    注册表配置全倒进来
        "frame_index": frame_index,                         #    再加"当前是第几帧"
    }
    if self.transforms is not None:
        result = self.transforms(result)                    # ⑥ 跑 Pipeline（生成 input_ids）
    return result                                           # ⑦ 返回"加工完"的 dict
```

（`dataset.py:50-62`）

**逐行讲（配 2.3 的例子，假设 `idx=4`）：**

| 行 | 在干什么 | 小白注解 |
| --- | --- | --- |
| ① | `self.sample_index[4]` → `(1, 0)`，一次拿到两个值 | Python 的**元组解包**：`a, b = (1, 0)` 相当于 `a=1; b=0` |
| ② | `id_to_jsonl[1]` → `"episode_0000001.jsonl"`，然后把**这个文件的所有行**读成一个字符串列表 | 注意读的是**整个文件**，不是只读那一行——因为后面要用到"别的帧" |
| ③ | `orjson.loads(lines[0])`：把第 0 行那个 **JSON 字符串**解析成 Python dict | `orjson` 就是"更快的 `json`"，用法一样；解析出来就是我们 2.1 里看到的那一行 |
| ④ | `result["raw_lines"] = lines`：把整份行列表**挂到样本上** | **必须挂**——`LoadHistory` 要往回看 20 个槽（最多回看 500 帧）、`BuildAction` 要往后取 50 帧，它们都得去访问"别的行"。只给当前行不够用 |
| ⑤ | 拼 `meta_data`：`**` 把注册表配置（`robot_type` / `speed` / `fps` …）**展开**进来，再补一个 `frame_index` | `**dict` 是"键值对展开"；`dataset_meta or {}` 是**兜底**：万一是 `None`，用空 dict 顶上，否则 `**None` 会直接报错。`frame_index` 是给 `LoadHistory`/`BuildAction` 算偏移用的（"从第 123 帧往回数 25 帧"） |
| ⑥ | 把 dict 交给 Pipeline（第 3～9 节讲的那 7 个 transform） | **这一步之后，dict 里才有 `input_ids` / `pixel_values` / `history_mask`** |
| ⑦ | 返回 | 返回的不是"原始 JSONL 行"，而是**跑完整条管线后的成品 dict** |

**一张图看清楚"谁在什么时候被调用"：**

```text
DataLoader（batch_size=B，多进程）
   │  对每个样本调一次 __getitem__(idx)
   ▼
JsonlDataset.__getitem__(idx)
   ├─ 查花名册 → (文件编号, 行号)
   ├─ 读文件 → 解析那一行 → 得到原始 dict（prompt/images/state/action）
   ├─ 挂 raw_lines + meta_data
   └─ transforms（Pipeline）：BuildAction → … → PadAction
   ▼
成品 dict（含 input_ids [1,L]、pixel_values [3,3,448,448]、history_mask …）
   │
   ▼
TrainingCollator 把 B 份成品拼成 batch → input_ids [B, 1536]
```

> ⚠️ **性能小提醒**：`_read_jsonl_lines` 没有缓存（`dataset.py:66-70` 就是直接 `open().readlines()`），所以**每取一个样本都要把整个文件重读一遍**。这是这套实现"简单换性能"的地方——换来的是 `raw_lines` 随手可用，不用额外维护"按帧号索引的内存缓存"。

> 💡 **读代码小贴士**：`**` 在 Python 里有两个完全不同的用途——函数调用时 `f(**kwargs)` 是"把字典展开成具名参数"，字典字面量里 `{**a, **b}` 是"把多个字典合并"。这里用的是后者。

---

## 3. 第 1 步：五个"备料" transform（都不产生 token）

RoboDojo mem SFT 的 Pipeline（`playground/dm05_mem_sft_robodojo_cover_blocks.py:96-127`）：

```python
Pipeline([
    self._action_transform(action_horizon),   # ① BuildAction：造未来 50 帧动作
    LoadImages(image_keys=…, image_dir=…),    # ② 读 3 路当前图
    LoadHistory(image_key=image_keys[0], …),  # ③ 读历史主视角帧
    PixelTransform(transform_pipeline=…),     # ④ 图像增强
    Normalize(norm_stats_path=…, …),          # ⑤ state / action 归一化
    ChatTokenization(processor=…, …),         # ⑥ ★ 生成 input_ids
    PadAction(32),                            # ⑦ 动作维度补齐到 32
])
```

| transform | 输入 | 输出（塞进 `data`） | 备注 |
| --- | --- | --- | --- |
| `BuildAction` | `raw_lines` + `frame_index` | `data["action"]`、`data["action_mask"]` | 组合了 `BuildActionChunk`（+ 相对动作模式下的 `ActionRelative`），`transforms.py:406` |
| `LoadImages` | `data["images_1..3"]` 的 url | `data["images"]`：3 张 PIL | 支持 `image` / `video` 两种类型，`transforms.py:199-226` |
| `LoadHistory` | `raw_lines` + `fps` | `data["history_images"]`：0～20 张 PIL | 见下 |
| `PixelTransform` | 3 张当前图 | 增强后的图 | 训练期数据增强（`p=0.5`） |
| `Normalize` | `state` / `action` | 归一化后的 `state` / `action` | `Normalize` 之后 `state` 才落在 `[-1,1]`，后面分桶才有意义 |

### 3.1 `LoadHistory` 的挑图规则（值得单独说）

```python
for slot in range(self.max_history_images, 0, -1):          # slot = 20, 19, …, 1
    raw_index = frame_index - int(round(slot / self.uniform_fps * source_fps))
    if raw_index < 0 or raw_index >= len(lines):
        continue                                            # 越界 → 丢弃
    history_images.append(load(...))
```

（`transforms.py:249-268`）

- `uniform_fps = 1.0`、`source_fps = fps = 25` → **每个槽回看 25 帧（1 秒）**，第 20 槽回看 500 帧（20 秒前）；
- 循环是 `20 → 1`（从最老到最新），所以 `history_images` 天然是 **oldest → newest**；
- 越界的槽**直接丢掉**，所以列表长度不固定（0～20）；
- **只取主视角**：`image_key=image_keys[0]`（= `images_1` = Head）。

> 这就是"为什么只有 Head 相机有历史"的答案——不是设计上的限制，而是 `LoadHistory` 只读了一路。

**到 `ChatTokenization` 门口时，`data` 里有**：`prompt`（字符串）、`images`（3 张 PIL）、`history_images`（0～20 张 PIL）、`state`（归一化向量）、`action`、`meta_data`。

---

## 4. 第 2 步：`ChatTokenization` 拼出 `messages`（★核心）

这是整条链路上"信息量最大"的一步。它做四件事（`transforms.py:716-782`）：

### 4.1 文本头：从 `meta_data` 拼字符串

```python
meta = data.get("meta_data", {})
text_parts = []
if meta.get("robot_type") is not None:
    text_parts.append(f"Robot: {meta['robot_type']}\n")
if meta.get("control_mode") is not None:                       # 有才写
    control_mode = "end effector" if meta["control_mode"] in {"eef","ee","end effector"} else meta["control_mode"]
    text_parts.append(f"Control mode: {control_mode}\n")
speed = meta.get("speed") or "0.5"                             # 缺省兜底
text_parts.append(f"Overall speed: {speed}\n")
text_parts.append(f"Task: {data['prompt']}.\n")                # 注意末尾加了一个句号

user_content = [{"type": "text", "text": "".join(text_parts)}]
```

**回答一个常见疑问**：`Robot: Aloha` / `Overall speed: 0.5` **不是数据集里原样存在的字符串**，而是这里用注册表配置 + 任务文本拼出来的模板文本：

- `Aloha` ← `RobotType.ALOHA` 的枚举值（注册表 `robot_type`）；
- `0.5` ← 注册表 `speed`，没有就兜底 `"0.5"`；
- `Control mode:` 只有配置里有才出现（所以有些渲染样例里看不到这一行）。

### 4.2 历史图槽：纯文本占位符

```python
history_images = [img for img in (data.get("history_images") or [])[-self.max_history_images:] if img is not None]
n_valid = len(history_images)
user_content[-1]["text"] += "History images: "
user_content[-1]["text"] += (
    "<unused1>" * (HISTORY_TOKENS_PER_IMAGE * (self.max_history_images - n_valid))   # 空槽
    + (("<unused0>" * HISTORY_TOKENS_PER_IMAGE) + "\n") * n_valid                    # 有效槽 + 换行
)
```

- `HISTORY_TOKENS_PER_IMAGE = 16`，`max_history_images = 20` → **payload 恒为 320 个 token**（见第 6 节的账）；
- 空槽用 `<unused1>`（id 7）、有效槽用 `<unused0>`（id 6），每个有效槽后补一个 `\n`；
- **空槽在左、有效槽在右**（训练管线的摆法）；推理的 `logical_step` 模式是逐槽摆放，空槽可能夹在中间；
- 同时算好图像张量：`history_pixel_values = processor.image_processor(images=..., return_tensors="pt")["pixel_values"]`（形状 `[H, 3, 448, 448]`）。

### 4.3 三路当前图：标签文本 + image item 交替

```python
for prompt, image in zip(self.image_prompts, data["images"], strict=True):
    user_content[-1]["text"] += f"{prompt} image: "   # "Head image: "
    user_content.append({"type": "image", "image": image})
```

（`transforms.py:761-767`）注意实现细节：**label 会追加到"上一个 content item"里**——如果上一个恰好是 image，就新开一个 text item。所以最终 `content` 是 `[text, image, text, image, text, image, text]` 这种交替结构。

`image_prompts` 来自注册表：`["Head", "Left wrist", "Right wrist"]`，和 `image_keys` 一一对应。

### 4.4 States：向量 → 分桶整数 → 普通文本

```python
if self.add_state:
    user_content.append({
        "type": "text",
        "text": "States: " + " ".join(str(b) for b in self.action_to_bin_tokens(data["state"], self.n_bins)),
    })
```

分桶函数（`transforms.py:700-714`）：

```python
def action_to_bin_tokens(self, action, n_bins=256):
    clipped = np.clip(action, -1.0, 1.0)          # 归一化后应在 [-1,1]
    normalized = (clipped + 1.0) / 2.0            # → [0,1]
    bins = np.floor(normalized * (n_bins - 1)).astype(int)
    return np.clip(bins, 0, n_bins - 1).tolist()  # → [0,255] 的整数
```

所以 14 维 state 变成 14 个 `0~255` 的整数，用空格连起来当**普通文本**喂进去（不占特殊结构）。

### 4.5 拼成 `messages`

```python
messages = [{"role": "user", "content": user_content}]
```

（`transforms.py:782`）至此，`messages` 里**没有图片本身**，只有 `{"type":"image","image":PIL}` 这样的"位置标记"——真正的展开在下一步。

---

## 5. 第 3 步：`apply_chat_template` 渲染 + 分词

```python
inputs = self.processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
```

（`transforms.py:783-789`）

### 5.1 分词器/处理器是什么

`AutoProcessor.from_pretrained(权重目录)`（`dm05_exp.py:610`）。实测（直接读权重目录的配置）：

| 项目 | 实测值 |
| --- | --- |
| `processor_class` | `Gemma3Processor` |
| `tokenizer_class` | `GemmaTokenizer`（Gemma 3 那套 SentencePiece 系） |
| `tokenizer.json` 基础词表 | 262144 |
| 模型侧 `vocab_size` | 262208 |
| `image_seq_length` | 256（在 `processor_config.json`） |
| `boi_token` / `eoi_token` / `image_token` | `<start_of_image>` / `<end_of_image>` / `<image_soft_token>` |

### 5.2 实测特殊 token id 表

| token | id | 作用 |
| --- | ---: | --- |
| `<pad>` / `<eos>` / `<bos>` | 0 / 1 / 2 | 填充 / 结束 / 开头 |
| `<unused0>` | **6** | 有效历史槽占位符（会被视觉特征覆写） |
| `<unused1>` | **7** | 空历史槽占位符（会被清零 + 屏蔽） |
| `<start_of_turn>` / `<end_of_turn>` | 105 / 106 | 对话回合边界 |
| `<start_of_image>`（boi） | 255999 | 图片块开始 |
| `<end_of_image>`（eoi） | 256000 | 图片块结束 |
| `<image_soft_token>` | **262144** | 当前图 soft token 占位符（= `image_token_id`） |

### 5.3 `chat_template.jinja` 干了什么

模板随权重一起分发，逻辑可以概括成：

```text
{{ bos_token }}                              → <bos>
<start_of_turn>user\n                        → 用户回合开始
   …content…                                 → 文本原样（每个文本块做 | trim）、图片位置输出一个 <start_of_image>
<end_of_turn>\n                              → 用户回合结束
<start_of_turn>model\n                       → add_generation_prompt=True 才追加
```

**两个容易看漏的细节**：

1. 模板对每个文本块 `| trim`——所以 `Head image: ` 末尾的空格在**实际渲染时会被去掉**（很多图解里为了对齐好看保留了空格，别当成真实 token）；
2. 模板对每张图**只写一个** `<start_of_image>`，**不写那 256 个 soft token**——那 256 个是 processor 按 `image_seq_length = 256` 展开出来的：

```text
模板输出：        <start_of_image>
processor 展开后：<start_of_image> + <image_soft_token>×256 + <end_of_image>   = 258 个座位
```

### 5.4 步 3 的输出

```python
inputs["input_ids"]      # [1, L]
inputs["attention_mask"] # [1, L]
inputs["token_type_ids"] # [1, L]
inputs["pixel_values"]   # [3, 3, 448, 448]
```

---

## 6. 第 4 步：算一遍 token 账（可自查）

前缀长度可以精确拆成四块（RoboDojo `cover_blocks`）：

```text
前缀长度 = 129（文本头 + 对话模板 + States，固定）
         + 320（历史槽 payload = 20 槽 × 16，固定）
         + K  （K = 有效历史图数，每张多一个换行符 ⏎）
         + 774（当前 3 视图 = 3 × (1 + 256 + 1)，固定）
```

| K（有效历史图数） | 129 | +320 | +K | +774 | 总长 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 129 | 320 | 0 | 774 | **1223** |
| 3 | 129 | 320 | 3 | 774 | **1226** |
| 20 | 129 | 320 | 20 | 774 | **1243** |

**三个真机样本的实测**（原文档 3.2 节）：

| 样本 | `input_ids` | `history_pixel_values` | `history_mask` True 数 | `visible prefix` |
| --- | --- | --- | ---: | ---: |
| `idx=0, frame=0` | `[1, 1213]` | `None` | 0 | 893 |
| `idx=400, frame=400` | `[1, 1232]` | `[16, 3, 448, 448]` | 256 = 16×16 | 1168 |
| `idx=1000, frame=445` | `[1, 1234]` | `[17, 3, 448, 448]` | 272 = 17×16 | 1186 |

**两个读表要点**：

1. 总长不是常数：**不同任务的 `Task:` 文本长短不同**，所以总数在 **1213～1243** 之间浮动（"129"是那条 cover_blocks 提示词的值，换任务就变）；但 320 / 774 是结构性的恒定值；
2. `visible prefix = 总长 − 空槽数×16`——因为空槽 `<unused1>` 会被注意力屏蔽，对模型"可见"的只有有效 token。验算：`1213 − 320 = 893` ✓，`1232 − 64 = 1168` ✓（16 张历史时空槽 = 4×16 = 64），`1234 − 48 = 1186` ✓。

> 所以"历史图从 0 张到 20 张只涨 20 个 token"这个反直觉的结论，来源就是：**空槽和有效槽的 payload 一样大（都是 16 个 token），换一个槽只多了一个换行**。

---

## 7. 第 5 步：超长时的特殊截断

如果 `L > max_length`（RoboDojo 配置是 **1536**，`playground/...:91`），**不是直接砍 token**（那会把某张图的 258 个座位拦腰截断），而是"改文本、重渲染"（`transforms.py:802-832`，下面是简化后的代码）：

```python
if inputs["input_ids"].shape[1] > self.max_length:
    prompt_for_truncation = data["prompt"]
    prompt_token_ids = self.tokenizer.encode(prompt_for_truncation, add_special_tokens=False)
    overflow = inputs["input_ids"].shape[1] - self.max_length
    keep_tokens = max(0, len(prompt_token_ids) - overflow - 16)     # 多留 16 个 token 缓冲

    if keep_tokens < len(prompt_token_ids):
        shortened_prompt = self.tokenizer.decode(prompt_token_ids[:keep_tokens], skip_special_tokens=False).strip()
        # 把 messages 里那段任务文本替换成缩短版
        for content_item in messages[0]["content"]:
            if content_item.get("type") != "text" or prompt_for_truncation not in content_item.get("text", ""):
                continue
            content_item["text"] = content_item["text"].replace(prompt_for_truncation, shortened_prompt, 1)
            break
        inputs = self.processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")
```

**为什么这么设计**：图片块、历史槽块都是"结构化"的，砍进中间会破坏结构（比如只剩半张图的 soft token）；而任务文本是**连续的自然语言**，缩短它最安全。

> ⚠️ 一个读代码时值得知道的细节：截断分支里那次**重新渲染漏了 `add_generation_prompt=True`**（推理侧同名逻辑是带的，`third_party/robochallenge_inference/policies/opendm_policy.py:124-130`），所以被截断的样本会少掉结尾的 `<start_of_turn>model\n`。因为前缀不参与 loss，影响有限。

---

## 8. 第 6 步：两个后处理数组

```python
token_type_ids = inputs["token_type_ids"]
if self.is_history:
    history_mask = inputs["input_ids"] == self.history_placeholder_token_id   # id 6，即 <unused0>
    token_type_ids = token_type_ids.clone()
    token_type_ids[history_mask] = 1        # 把历史槽也算成"图像 token"
```

（`transforms.py:833-838`）

- `history_mask`：**不是模型算的，是这里就地算出来的**——因为 `<unused0>` 的 id 是 6，一句比较就得到掩码，后面模型内部靠它做 `masked_scatter`；
- `token_type_ids[history_mask] = 1`：让每张历史图的 16 个 token 在 Gemma3 的注意力掩码里组成一个**双向块**（和真正的图像块同等待遇）。

---

## 9. 第 7 步：`TrainingCollator` 补齐 + 拼 batch

```text
# 伪代码：collator 对每个样本做的事
max_len = self.max_length                    # = 1536
for 每个样本:
    if seq_len > max_len: 截断（并 warning）
    pad_len = max_len - seq_len
    input_ids      → 右补 pad_token_id (0)
    attention_mask → 右补 0
    token_type_ids → 右补 0
# 历史相关字段
history_mask   → 右补 False（或截到等长）
history_pixel_values → 沿 dim 0 torch.cat
# 最后
input_ids / attention_mask / token_type_ids / pixel_values / action / action_mask 都沿 batch 维 cat
```

（`collator.py:16-122`）

**注意两点**：

1. 补齐目标是**固定的 1536**，不是"batch 内最长那条"——这样 batch 形状恒定，CUDA Graph / 显存都友好；
2. `history_pixel_values` 是**沿 dim 0 直接 cat**，所以它是 `[ΣHᵢ, 3, 448, 448]` 而不是 `[B, H, 3, 448, 448]`。这不影响正确性：`masked_scatter` 对源张量和掩码都是扁平化处理的，而样本顺序与掩码的行主序一致，"第 k 张图 → 第 k 个有效槽"依然对得上。

---

## 10. 完整代码索引（想深挖看这里）

| 想了解 | 位置 |
| --- | --- |
| 样本索引构建（`max(0, 行数-1)`） | `opendm/data/dataset.py:25-39` |
| `index_cache.json` 生成 | `opendm/data/dataset.py:81-103` |
| `__getitem__`（JSONL + meta_data 合并） | `opendm/data/dataset.py:50-62` |
| 数据集注册表（`CONVERSATION_DATA`） | `opendm/dataset/register.py`、`opendm/dataset/robodojo_ee.py` |
| Pipeline 定义 | `opendm/data/transforms.py:33-46` |
| RoboDojo mem SFT 的 pipeline 顺序 | `playground/dm05_mem_sft_robodojo_cover_blocks.py:96-127` |
| `LoadImages` / `LoadHistory` | `opendm/data/transforms.py:199-226` / `229-268` |
| 分桶函数 `action_to_bin_tokens` | `opendm/data/transforms.py:700-714` |
| **`ChatTokenization.__call__`（全文主角）** | `opendm/data/transforms.py:716-850` |
| 超长截断 + 重渲染 | `opendm/data/transforms.py:802-832` |
| `history_mask` / `token_type_ids` 后处理 | `opendm/data/transforms.py:833-838` |
| `TrainingCollator` | `opendm/data/collator.py:16-122` |
| processor 加载 | `opendm/exp/dm05_exp.py:610` |
| `tokenizer_max_length=1536`（RoboDojo） | `playground/dm05_mem_sft_robodojo_cover_blocks.py:91` |
| 推理侧的同名逻辑（对照用） | `third_party/robochallenge_inference/policies/opendm_policy.py:36-145` |

---

## 11. 常见疑问

**Q1：为什么 pad 到 1536，而不是补到 batch 内最长？**
形状恒定。推理侧要把前缀录进 CUDA Graph（固定 bucket），训练侧 batch 形状恒定也更省事。实测最长约 1243 < 1536，所以正常情况下不会触发截断。

**Q2：`<unused1>`（空历史槽）在 `attention_mask` 里为什么是 1？**
因为它在"要不要算注意力"这层是有效的（它就是 prompt 的一部分）。真正把它变"隐形"的是**模型内部**的两道处理：`mask_history_pad_tokens_in_attention` 把 attention mask 置 0，以及 `inputs_embeds[input_ids == 7] = 0` 把 embedding 清零。

**Q3：同一个 prompt 在训练和推理时一样吗？**
结构一样、代码是两份（`transforms.py` vs `opendm_policy.py`），差别在：推理用 `logical_step` 模式逐槽摆放历史占位符，且 `speed` / `control_mode` 常来自运行时参数。这也是"训练/推理要逐项对齐"最容易出问题的点之一。

**Q4：`history_mask` 为什么要在数据侧算，模型不能自己找 `<unused0>` 吗？**
能，而且解码后 `<unused0>` 的语义就是"这里等下要接历史特征"。单独算一个 bool 数组是为了让模型侧少做一次比较、也方便 collator 统一补齐。

**Q5：`Task:` 文本被截断了会不会影响训练？**
会丢一点任务信息（截掉的是句尾），但换来的是"prompt 结构完整 + 长度可控"。实测样本都远小于 1536，正常训练几乎不会触发。

**Q6：图片为什么不直接塞进 `input_ids` 的文本里？**
因为它根本不是文本。`input_ids` 里只有**占位符的 id**；图片内容是在模型内部用 `masked_scatter` 把视觉向量**整行覆写**到这些座位上（见配套文档 1.5 节）。

**Q7：为什么历史槽是 16 个 token，而当前图是 256 个？**
当前图 256 = 16×16（视觉塔 2×2 池化后的产物）；历史图在这基础上**再做一次 4×4 池化**，压到 16 个——用精度换前缀长度（20 张历史图若按 256/图就是 5120 个 token）。

---

## 12. 相关文档

- `DM05模型结构与训练推理详解_小白版.md` —— 1.5 视觉塔与历史图处理、1.6 前缀 token 布局、2.2 数据 → 张量
- `模型架构与推理全流程.md` —— 速查手册（含 3.2 节真机样本实测、3.3 节 prompt 渲染）
- `starvla/docs/zerith_docs/training-RTC/02_原理_PI0与PI05前向推理.md` —— π0.5 侧的同类数据管线（可对照）
