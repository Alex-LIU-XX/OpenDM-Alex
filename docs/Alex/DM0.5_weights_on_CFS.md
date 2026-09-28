# DM0.5 权重存放位置（速查）

- 更新时间：2026-09-21
- **权重路径：`/mnt/cfs/opendm/checkpoints/`** ← CFS 上的**唯一副本**
- 本地 `/root/workspace/opendm/checkpoints/DM05` **已删除**（2026-09-21，释放 11 G）；要跑实验请用上面的 CFS 路径，或先按下文恢复

## 里面有什么

| 路径 | 内容 |
| --- | --- |
| `/mnt/cfs/opendm/checkpoints/DM05/` | 权重本体：`model.safetensors`（11.66 GB）+ `config.json`、`norm_stats.json`、tokenizer 等共 12 个文件 |
| `/mnt/cfs/opendm/checkpoints/DM05-sdpa-verify/` | 验证用覆盖层：`config.json` 为实体，其余 11 个是指向 `../DM05/` 的符号链接 |
| `/mnt/cfs/opendm/checkpoints/.hf_home/` | HF hub ref 缓存（28 KB，可忽略） |

- 来源：HuggingFace `Dexmal/DM05`，revision `a7bf3f465de915a44ed9a4ac6c403ab3466f2735`
- 主权重 sha256：`b7da77f516ebd0c0c68faed2f0cfe1c68985b7b874e49dda4292d8d286de20dd`（与 Hub LFS oid 一致）
- 规格：3 路图像 / 14 维 state-action / chunk 50；`norm_stats.json` 含 `DOS W1`（默认）与 `Aloha` 两个 profile

## 怎么用

```bash
cd /root/workspace/opendm
script/dm05_launcher.sh \
  --exp opendm/exp/dm05_exp.py \
  --task inference \
  --model-config.model-name-or-path /mnt/cfs/opendm/checkpoints/DM05 \
  --model-config.chunk-size 50 \
  --inference-config.output-action-dim 14 \
  --inference-config.image-prompts "Head" "Left wrist" "Right wrist" \
  --inference-config.port 7891
```

> 仓库文档里默认写的 `./checkpoints/DM05` 已不存在，**不要再照抄**，把路径换成上面的 CFS 绝对路径即可。训练/SFT 的 `--model-config.model-name-or-path`、`script/libero_runner.sh model --model-dir` 同理。

## 恢复到本地 / 只建软链

```bash
# 方式一：恢复实体副本（约 11 G，训练走本地盘更快）
rsync -a --partial --info=progress2 \
  /mnt/cfs/opendm/checkpoints/DM05/ /root/workspace/opendm/checkpoints/DM05/

# 方式二：只建软链，保持 ./checkpoints/DM05 路径可用（走 NFS I/O）
ln -s /mnt/cfs/opendm/checkpoints/DM05 /root/workspace/opendm/checkpoints/DM05
```

## 注意事项

- **唯一副本，别删**。可用 `sha256sum /mnt/cfs/opendm/checkpoints/DM05/model.safetensors` 自检。
- CFS 由平台挂载：`cfs-pVlRwhavQO.lb-2etad1gk.cfs.bj.baidubce.com:/` → `/mnt/cfs`（nfs4.1、rw）。容器重建后若 `/mnt/cfs` 不存在，按此 server/export 重新挂载（镜像内没有 `nfs-common`，需先装）。
- 在 CFS 上**不要做全盘 `find`/`du`**（元数据遍历很慢，会超时）；顺序读约 248 MB/s，小目录树操作正常。
- 写 CFS 需要会话文件策略允许：`workspace-write` 下写会被沙箱拒，`danger-full-access` 可写。

> 相关文档：[权重与 Benchmark 对应关系](DM0.5_weights_and_benchmark_mapping.md)、[环境配置与冒烟测试记录](opendm_env_setup_and_smoke_test.md)
