import logging
import torch
import torch.nn as nn
import torch.utils.benchmark

from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM, LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding, LlamaRMSNorm
from utils import replace_linear_with_custom

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

class CustomLlamaConfig(LlamaConfig):
    model_type = "custom_llama"

    def __init__(
        self,
        layer_configs=None,
        method=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer_configs = layer_configs
        self.method = method

class CustomLlamaModel(LlamaModel):
    config_class = CustomLlamaConfig

    def __init__(self, config: CustomLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        replace_linear_with_custom(
            self.layers,
            config.layer_configs,
            config.method
        )

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

class CustomLlamaModelForCausalLM(LlamaForCausalLM):
    config_class = CustomLlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = CustomLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

def benchmark(method, framework, layer_configs, num_seq, num_iters, device, dtype):
    text = "Your input text goes here ..."
    config = CustomLlamaConfig.from_pretrained("huggyllama/llama-7b", method=method, layer_configs=layer_configs)
    model = CustomLlamaModelForCausalLM.from_pretrained("huggyllama/llama-7b", config=config).to(device).to(dtype)
    model = torch.compile(model, fullgraph=True, mode='reduce-overhead')

    tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
    tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer(text, return_tensors="pt", max_length=num_seq, truncation=True, padding="max_length")
    tokens = {key: value.to(device) for key, value in tokens.items()}

    print(f"Benchmarking end-to-end llama7b {method} {framework} prefill with {num_seq} sequence length ...")

    def _time_function(model, tokens):
        with torch.no_grad():
            outputs = model(**tokens)
        torch.cuda.synchronize()

    # Warm-Up (Cold Start)
    for _ in range(10):
        with torch.no_grad():
            outputs = model(**tokens)

    t = torch.utils.benchmark.Timer(
        stmt='_time_function(model, tokens)',
        globals={'_time_function': _time_function, 'model': model, 'tokens': tokens},
        num_threads=torch.get_num_threads(),
    )

    temp = t.timeit(num_iters)
    print(temp)
    return temp
