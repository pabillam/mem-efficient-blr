import logging
import torch
import torch.nn as nn
import torch.utils.benchmark
from triton.testing import do_bench
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from utils import replace_linear_with_custom

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

NSYS = False

class CustomGPT2SConfig(GPT2Config):
    model_type = "gpt2"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        layer_configs=None,
        method=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer_configs = layer_configs
        self.method = method
    
class CustomGPT2SModel(GPT2Model):
    _supports_param_buffer_assignment = False

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        replace_linear_with_custom(
            self.h,
            config.layer_configs,
            config.method
        )
        
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation

        self.post_init()

def benchmark(method, framework, layer_configs, num_seq, num_iters, device, dtype):
    text = "Your input text goes here ..."
    config = CustomGPT2SConfig.from_pretrained("openai-community/gpt2", method=method, layer_configs=layer_configs)
    model = CustomGPT2SModel.from_pretrained("openai-community/gpt2", config=config).to(device).to(dtype)
    model = torch.compile(model, fullgraph=True, mode='reduce-overhead')

    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer(text, return_tensors="pt", max_length=num_seq, truncation=True, padding="max_length")
    tokens = {key: value.to(device) for key, value in tokens.items()}
    print(f"Benchmarking end-to-end gpt2s {method} {framework} prefill with {num_seq} sequence length ...")

    def _time_function(model, tokens):
        with torch.no_grad():
            outputs = model(**tokens)
        torch.cuda.synchronize()
    
    # Warm-Up (Cold Start)
    for _ in range(10):
        with torch.no_grad():
            outputs = model(**tokens)

    if NSYS:
        for i in range(20):
            if i == 10: torch.cuda.cudart().cudaProfilerStart()
            if i >= 10: torch.cuda.nvtx.range_push(f"GPT2-Small {method} {framework} Iteration {i}")
            _time_function(model, tokens)
            if i >= 10: torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()

        temp = 0
    else:
        t = torch.utils.benchmark.Timer(
            stmt='_time_function(model, tokens)',
            globals={'_time_function': _time_function, 'model': model, 'tokens': tokens},
            num_threads=torch.get_num_threads(),
        )

        temp = t.blocked_autorange(min_run_time=5.0)
        print(temp)

    return temp
