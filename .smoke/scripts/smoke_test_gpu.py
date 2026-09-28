"""OpenDM GPU 冒烟测试：把微型 DM05 放到 CUDA 上跑通训练前向/反向与动作推理。

- 训练路径：bf16_mixed policy + torch.autocast("cuda", bf16)
- 推理路径：eval + suffix attention backend = sdpa → 触发 CUDA Graph capture/replay 分支
"""

import traceback

import torch

try:
    from transformers.models.gemma3.configuration_gemma3 import (
        Gemma3Config,
        Gemma3TextConfig,
    )
except ImportError:
    from transformers import Gemma3Config, Gemma3TextConfig

try:
    from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
except ImportError:
    from transformers import SiglipVisionConfig

import copy

from opendm.constants.precision import BF16_MIXED_PRECISION_POLICY
from opendm.model.dm05.dm05_arch import DM05Config, DM05ForConditionalGeneration

TEXT_KWARGS = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=1000,
    max_position_embeddings=512,
    pad_token_id=0,
    sliding_window=256,
)
VISION_KWARGS = dict(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    image_size=64,
    patch_size=16,
    num_channels=3,
)
ACTION_DIM, CHUNK_SIZE, SEQ_LEN, BATCH = 8, 4, 12, 1
DEVICE = "cuda"


def build_tiny_model() -> DM05ForConditionalGeneration:
    language_config = Gemma3TextConfig(**TEXT_KWARGS)
    vision_config = SiglipVisionConfig(**VISION_KWARGS)
    vlm_config = Gemma3Config(
        text_config=language_config, vision_config=vision_config
    )
    config = DM05Config(
        vlm_config=vlm_config,
        action_config=copy.deepcopy(vlm_config.text_config),
        action_dim=ACTION_DIM,
        chunk_size=CHUNK_SIZE,
        precision_policy=BF16_MIXED_PRECISION_POLICY,
    )
    return DM05ForConditionalGeneration(config).to(DEVICE)


def make_inputs():
    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(1, 900, (BATCH, SEQ_LEN), generator=g).to(DEVICE)
    attention_mask = torch.ones_like(input_ids)
    action = torch.randn(BATCH, CHUNK_SIZE, ACTION_DIM, generator=g).to(DEVICE)
    action_mask = torch.ones(BATCH, CHUNK_SIZE, 1, dtype=torch.bool, device=DEVICE)
    return input_ids, attention_mask, action, action_mask


def check(name, fn):
    print(f"\n--- {name} ---", flush=True)
    try:
        result = fn()
    except Exception:
        print(f"[FAIL] {name}", flush=True)
        traceback.print_exc()
        return False
    print(f"[PASS] {name}: {result}", flush=True)
    return True


def test_gpu_forward_backward():
    model = build_tiny_model()
    model.train()
    input_ids, attention_mask, action, action_mask = make_inputs()
    # 纯文本 batch：token_type_ids 全 0（图像/历史位置才是 1，见 data/collator.py）
    token_type_ids = torch.zeros_like(input_ids)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            action=action,
            action_mask=action_mask,
        )
    assert torch.isfinite(out.loss), f"loss not finite: {out.loss}"
    out.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return (
        f"loss={out.loss.item():.4f}, params_with_grad={len(grads)}, "
        f"peak_mem={peak:.1f}MiB"
    )


def test_gpu_inference_action_graph():
    model = build_tiny_model()
    model.eval()
    # CUDA eval + sdpa 才会走 suffix CUDA Graph 分支（与真实推理默认一致）
    model.model.action_expert.set_action_attention_backend("sdpa")
    input_ids, attention_mask, _, action_mask = make_inputs()
    with torch.no_grad():
        first = model.inference_action(
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_mask=action_mask,
            diffusion_steps=3,
        )
        second = model.inference_action(
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_mask=action_mask,
            diffusion_steps=3,
        )
    assert tuple(first.shape) == (BATCH, CHUNK_SIZE, ACTION_DIM), first.shape
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    n_profiles = len(getattr(model, "_suffix_graph_profiles", {}))
    return (
        f"actions{tuple(first.shape)} mean={first.mean().item():.4f}, "
        f"graph_profiles={n_profiles} (首次 eager，第二次 capture/replay)"
    )


def test_gpu_eager_inference():
    model = build_tiny_model()
    model.eval()
    model.model.action_expert.set_action_attention_backend("eager")
    input_ids, attention_mask, _, action_mask = make_inputs()
    with torch.no_grad():
        actions = model.inference_action(
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_mask=action_mask,
            diffusion_steps=2,
        )
    assert torch.isfinite(actions).all()
    return f"actions{tuple(actions.shape)} mean={actions.mean().item():.4f}"


def main() -> int:
    import opendm

    print(f"opendm: {opendm.__file__}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("GPU 不可用，退出")
        return 1
    p = torch.cuda.get_device_properties(0)
    print(
        f"gpu: {p.name} cc{p.major}.{p.minor} mem={p.total_memory/1024**3:.1f}GiB "
        f"sm={p.multi_processor_count} bf16={torch.cuda.is_bf16_supported()}"
    )

    results = [
        check("GPU 训练 forward + backward (bf16 autocast)", test_gpu_forward_backward),
        check("GPU 动作推理 (sdpa + CUDA Graph 分支)", test_gpu_inference_action_graph),
        check("GPU 动作推理 (eager 回退分支)", test_gpu_eager_inference),
    ]
    print("\n==== GPU SMOKE RESULT ====")
    for ok, name in zip(results, ["forward_backward", "inference_sdpa_graph", "inference_eager"]):
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"GPU SMOKE {'OK' if all(results) else 'FAILED'}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
