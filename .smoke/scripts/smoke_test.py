"""OpenDM 冒烟测试：随机初始化一个微型 DM05（CPU，FP32），验证前向与动作推理能跑通。

不依赖任何预训练权重，纯结构/数值层面的 smoke test。
"""

import copy
import traceback

import torch

from opendm.constants.precision import FP32_MIXED_PRECISION_POLICY
from opendm.model.dm05.dm05_arch import DM05Config, DM05ForConditionalGeneration

try:
    from transformers.models.gemma3.configuration_gemma3 import (
        Gemma3Config,
        Gemma3TextConfig,
    )
except ImportError:  # 兼容不同 transformers 版本
    from transformers import Gemma3Config, Gemma3TextConfig

try:
    from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
except ImportError:
    from transformers import SiglipVisionConfig


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

ACTION_DIM = 8
CHUNK_SIZE = 4
SEQ_LEN = 12
BATCH = 1


def build_tiny_model() -> DM05ForConditionalGeneration:
    language_config = Gemma3TextConfig(**TEXT_KWARGS)
    vision_config = SiglipVisionConfig(**VISION_KWARGS)
    vlm_config = Gemma3Config(
        text_config=language_config, vision_config=vision_config
    )
    # action_config 必须与 VLM 的语言配置逐字段一致（validate_action_config_compatible），
    # 所以从构造完的 text_config 深拷贝，避免 Gemma3Config 内部改动导致不一致。
    action_config = copy.deepcopy(vlm_config.text_config)
    config = DM05Config(
        vlm_config=vlm_config,
        action_config=action_config,
        action_dim=ACTION_DIM,
        chunk_size=CHUNK_SIZE,
        precision_policy=FP32_MIXED_PRECISION_POLICY,
    )
    model = DM05ForConditionalGeneration(config)
    model.eval()
    return model


def make_text_inputs():
    generator = torch.Generator().manual_seed(0)
    # 避开 pad id(0) 与任何特殊 token id（vocab=1000，image token 远大于该值）
    input_ids = torch.randint(1, 900, (BATCH, SEQ_LEN), generator=generator)
    attention_mask = torch.ones_like(input_ids)
    action = torch.randn(BATCH, CHUNK_SIZE, ACTION_DIM, generator=generator)
    action_mask = torch.ones(BATCH, CHUNK_SIZE, 1, dtype=torch.bool)
    return input_ids, attention_mask, action, action_mask


def check(name: str, fn):
    print(f"\n--- {name} ---", flush=True)
    try:
        result = fn()
    except Exception:
        print(f"[FAIL] {name}", flush=True)
        traceback.print_exc()
        return False
    print(f"[PASS] {name}: {result}", flush=True)
    return True


def test_forward():
    model = build_tiny_model()
    input_ids, attention_mask, action, action_mask = make_text_inputs()
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        action=action,
        action_mask=action_mask,
    )
    assert out.loss is not None, "loss is None"
    assert torch.isfinite(out.loss), f"loss not finite: {out.loss}"
    out.loss.backward()
    grad_norms = {
        n: float(p.grad.norm())
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    assert grad_norms, "backward produced no gradients"
    assert all(
        v == v for v in grad_norms.values()
    ), "NaN gradient detected"
    return (
        f"loss={out.loss.item():.4f}, fm_loss={out.fm_loss.item():.4f}, "
        f"params_with_grad={len(grad_norms)}"
    )


def test_inference_action():
    model = build_tiny_model()
    input_ids, attention_mask, _, action_mask = make_text_inputs()
    with torch.no_grad():
        actions = model.inference_action(
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_mask=action_mask,
            diffusion_steps=3,
        )
    assert tuple(actions.shape) == (BATCH, CHUNK_SIZE, ACTION_DIM), actions.shape
    assert torch.isfinite(actions).all(), "non-finite action output"
    return f"actions{tuple(actions.shape)} mean={actions.mean().item():.4f}"


def test_param_summary():
    model = build_tiny_model()
    summary = model.model.parameter_summary()
    assert summary["total"] > 0
    assert "action_expert" in summary and "vlm" in summary
    return f"vlm={summary['vlm']}, action_expert={summary['action_expert']}, total={summary['total']}"


def main() -> int:
    import opendm

    print(f"opendm: {opendm.__file__}")
    print(f"torch: {torch.__version__} cuda_available={torch.cuda.is_available()}")

    results = [
        check("import + 微型模型构建", lambda: type(build_tiny_model()).__name__),
        check("训练 forward + backward", test_forward),
        check("inference_action 欧拉采样", test_inference_action),
        check("参数量统计", test_param_summary),
    ]
    print("\n==== SMOKE RESULT ====")
    for ok, name in zip(results, ["build", "forward", "inference_action", "param_summary"]):
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"SMOKE {'OK' if all(results) else 'FAILED'}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
