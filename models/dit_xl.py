import torch
import torch.nn as nn
import torch.utils.benchmark
import logging
from DiT.models import DiT_models
from utils import replace_linear_with_custom

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

class CustomDiT(nn.Module):
    def __init__(self, model_name="DiT-XL/2", layer_configs=None, method=None, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        img_size = 256
        self.patch_size = 2
        self.seq_len = (img_size // self.patch_size) ** 2

        self.model = DiT_models[model_name](input_size=img_size)
        self.model.eval()
        self.model = self.model.to(device).to(dtype)

        self.device = device
        self.dtype = dtype

        if layer_configs is not None and method is not None:
            replace_linear_with_custom(self.model.blocks, layer_configs, method)

    def forward(self, x):
        t = torch.zeros(x.size(0), device=self.device, dtype=torch.float32)
        y = torch.zeros(x.size(0), device=self.device, dtype=torch.long)
        return self.model(x, t, y)

def benchmark(method, framework, layer_configs, num_seq, num_iters, device, dtype):
    
    model = CustomDiT(model_name="DiT-XL/2", layer_configs=layer_configs, method=method, dtype=dtype, device=device)
    model = torch.compile(model, fullgraph=True, mode='reduce-overhead')

    images = torch.randn(1, 4, 256, 256, device=device, dtype=dtype)

    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = model(images)

    def _time_function(model, images):
        with torch.no_grad():
            _ = model(images)
        torch.cuda.synchronize()

    t = torch.utils.benchmark.Timer(
        stmt='_time_function(model, images)',
        globals={'_time_function': _time_function, 'model': model, 'images': images},
        num_threads=torch.get_num_threads(),
    )

    temp = t.timeit(num_iters)
    print(temp)
    return temp
