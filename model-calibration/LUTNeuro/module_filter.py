import re


QWEN3_MLP_RE = re.compile(r"^model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)$")
QWEN3_ATTN_RE = re.compile(r"^model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$")


def should_luterize_module(module_name: str, target: str) -> bool:
    if module_name == "lm_head" or module_name.endswith(".lm_head"):
        return False
    if target == "mlp":
        return QWEN3_MLP_RE.match(module_name) is not None
    if target == "attention":
        return QWEN3_ATTN_RE.match(module_name) is not None
    if target == "all":
        return (
            QWEN3_MLP_RE.match(module_name) is not None
            or QWEN3_ATTN_RE.match(module_name) is not None
        )
    raise ValueError(f"unsupported target module set: {target}")
