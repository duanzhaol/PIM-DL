import torch.nn as nn

from LUTNeuro.LUTLinear_t import LUTLinear_t
from LUTNeuro.LUTerize import LUTerize


class FakeSelfAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16, bias=False)


class FakeMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(16, 32, bias=False)
        self.up_proj = nn.Linear(16, 32, bias=False)
        self.down_proj = nn.Linear(32, 16, bias=False)


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeSelfAttn()
        self.mlp = FakeMlp()


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([FakeLayer()])
        self.lm_head = nn.Linear(16, 100, bias=False)


def test_luterize_replaces_qwen3_mlp_without_touching_attention_or_lm_head():
    model = FakeModel()
    luterizer = LUTerize(
        model=model,
        dataloader=None,
        tokenizer=None,
        logger=None,
        activation_dir=None,
        centroid_dir=None,
        output_dir=None,
        ncentroid=4,
        vec_len=4,
        init_centroids=False,
        target_modules="mlp",
    )

    luterizer.luterize_model()

    layer = model.model.layers[0]
    assert isinstance(layer.mlp.gate_proj, LUTLinear_t)
    assert isinstance(layer.mlp.up_proj, LUTLinear_t)
    assert isinstance(layer.mlp.down_proj, LUTLinear_t)
    assert isinstance(layer.self_attn.q_proj, nn.Linear)
    assert isinstance(model.lm_head, nn.Linear)
