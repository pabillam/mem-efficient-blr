import logging
import torch
import torch.nn as nn
import torch.utils.benchmark
from transformers import ViTModel, ViTConfig
from transformers.models.vit.modeling_vit import ViTLayer
from utils import replace_linear_with_custom

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

NSYS = False

class FusedQKVLinear(nn.Module):
    def __init__(self, embed_dim, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)

    def forward(self, hidden_states):
        qkv = self.qkv(hidden_states)
        q, k, v = qkv.split(self.embed_dim, dim=-1)
        return q, k, v

def fuse_qkv_in_vit(model):
    for layer in model.encoder.layer:
        attn = layer.attention.attention
        embed_dim = attn.query.in_features
        bias = attn.query.bias is not None

        fused = FusedQKVLinear(embed_dim, bias=bias).to(
            attn.query.weight.device, attn.query.weight.dtype
        )

        with torch.no_grad():
            fused.qkv.weight[:embed_dim].copy_(attn.query.weight)
            fused.qkv.weight[embed_dim:2*embed_dim].copy_(attn.key.weight)
            fused.qkv.weight[2*embed_dim:].copy_(attn.value.weight)

            if bias:
                fused.qkv.bias[:embed_dim].copy_(attn.query.bias)
                fused.qkv.bias[embed_dim:2*embed_dim].copy_(attn.key.bias)
                fused.qkv.bias[2*embed_dim:].copy_(attn.value.bias)

        for name in ["query", "key", "value"]:
            if hasattr(attn, name):
                delattr(attn, name)

        attn.add_module("qkv", fused)

        def forward_with_fused(self, hidden_states, *args, **kwargs):
            q, k, v = self.qkv(hidden_states)
            return q, k, v

        attn.forward = forward_with_fused.__get__(attn, attn.__class__)

class CustomViTConfig(ViTConfig):
    model_type = "vit"

    def __init__(
        self,
        layer_configs=None,
        method=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer_configs = layer_configs
        self.method = method


class CustomViTModel(ViTModel):
    _supports_param_buffer_assignment = False

    def __init__(self, config):
        super().__init__(config)

        fuse_qkv_in_vit(self)

        replace_linear_with_custom(
            self.encoder.layer,
            config.layer_configs,
            config.method,
            skip_keys=["attention.output.dense"]
        )

        self.post_init()

def benchmark(method, framework, layer_configs, num_seq, num_iters, device, dtype):
    config = CustomViTConfig.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        method=method,
        layer_configs=layer_configs,
    )
    model = CustomViTModel.from_pretrained(
        "google/vit-base-patch16-224-in21k", config=config
    ).to(device).to(dtype)
    model = torch.compile(model, fullgraph=True, mode="reduce-overhead")

    images = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)
    print(f"Benchmarking end-to-end vit-base {method} {framework} with 224x224 image ...")

    def _time_function(model, images):
        with torch.no_grad():
            outputs = model(pixel_values=images)
        torch.cuda.synchronize()

    # Warm-Up
    for _ in range(10):
        with torch.no_grad():
            outputs = model(pixel_values=images)

    if NSYS:
        for i in range(20):
            if i == 10:
                torch.cuda.cudart().cudaProfilerStart()
            if i >= 10:
                torch.cuda.nvtx.range_push(f"ViT-Base {method} {framework} Iteration {i}")
            _time_function(model, images)
            if i >= 10:
                torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
        temp = 0
    else:
        t = torch.utils.benchmark.Timer(
            stmt="_time_function(model, images)",
            globals={"_time_function": _time_function, "model": model, "images": images},
            num_threads=torch.get_num_threads(),
        )
        temp = t.blocked_autorange(min_run_time=5.0)
        print(temp)

    return temp
