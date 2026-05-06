from LUTNeuro.module_filter import should_luterize_module


def test_qwen3_mlp_target_matching():
    assert should_luterize_module("model.layers.0.mlp.gate_proj", "mlp")
    assert should_luterize_module("model.layers.35.mlp.up_proj", "mlp")
    assert should_luterize_module("model.layers.12.mlp.down_proj", "mlp")
    assert not should_luterize_module("model.layers.12.self_attn.q_proj", "mlp")
    assert not should_luterize_module("lm_head", "mlp")


def test_qwen3_attention_target_matching():
    assert should_luterize_module("model.layers.0.self_attn.q_proj", "attention")
    assert should_luterize_module("model.layers.0.self_attn.k_proj", "attention")
    assert should_luterize_module("model.layers.0.self_attn.v_proj", "attention")
    assert should_luterize_module("model.layers.0.self_attn.o_proj", "attention")
    assert not should_luterize_module("model.layers.0.mlp.gate_proj", "attention")


def test_qwen3_all_target_matching_excludes_lm_head():
    assert should_luterize_module("model.layers.1.mlp.down_proj", "all")
    assert should_luterize_module("model.layers.1.self_attn.o_proj", "all")
    assert not should_luterize_module("lm_head", "all")
