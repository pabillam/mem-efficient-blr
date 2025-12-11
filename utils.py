import yaml
from enum import Enum
import torch
import os
from models.wrappers import BlastLinear, MonarchLinear, DenseLinear, LowRankLinear
from transformers import Conv1D

class VerificationStatus(Enum):
    MATCH = 1
    DIFFER = 0
    FAIL = -1
    UNKNOWN = -2
    NA = -3

def next_power_of_2(x):
    if x < 1:
        return 1
    return 1 << (x - 1).bit_length()

def load_yaml(file_path):
    """ Load a YAML file and return its contents as a dictionary """
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def merge_dicts(dict1, dict2):
    for key, value in dict2.items():
        if isinstance(value, dict) and key in dict1:
            dict1[key] = merge_dicts(dict1.get(key, {}), value)
        elif isinstance(value, list) and key in dict1:
            dict1[key] = list(set(dict1[key] + value))
        else:
            dict1[key] = value
    return dict1

def do_verify(triton_out, torch_out, rtol=0, atol=1e-1):
    if torch.allclose(triton_out, torch_out, atol=atol, rtol=rtol):
        print("✅ Triton and Torch match")
        return VerificationStatus.MATCH
    else:
        print("❌ Triton and Torch differ")
        print(f"Max Absolute Difference Triton vs Torch = {torch.max(torch.abs(triton_out) - torch.abs(torch_out))}")
        print(f"Max Triton Output = {torch.max(triton_out)}")
        print(f"Min Triton Output = {torch.min(triton_out)}")
        return VerificationStatus.DIFFER

def replace_linear_with_custom(model, layer_configs, method, skip_keys=None):
    for name, module in model.named_modules():
        for key, config in layer_configs.items():
            if key in name and (isinstance(module, torch.nn.Linear) or isinstance(module, Conv1D)):
                skip_keys = skip_keys or []
                
                if any(skip in name for skip in skip_keys):
                    continue
                
                parent_name = ".".join(name.split(".")[:-1])
                attr_name = name.split(".")[-1]

                parent_module = model
                if parent_name:
                    for part in parent_name.split("."):
                        parent_module = getattr(parent_module, part)

                if isinstance(module, torch.nn.Linear):
                    in_features, out_features = module.in_features, module.out_features
                elif isinstance(module, Conv1D):
                    in_features, out_features = module.nx, module.nf

                if method == "blast" or method == "blast_sym_quant":
                    new_layer = BlastLinear(
                        in_features=in_features,
                        out_features=out_features,
                        b=config["b"],
                        rank=config["rank"],
                        func_path=config["func"],
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                        bias=module.bias is not None,
                        is_compiled=config.get("compile", False)
                    )
                elif method == "monarch":
                    new_layer = MonarchLinear(
                        in_features=in_features,
                        out_features=out_features,
                        b=config["b"],
                        rank=config["rank"],
                        func_path=config["func"],
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                        bias=module.bias is not None,
                        is_compiled=config.get("compile", False)
                    )
                elif method == "low_rank":
                    new_layer = LowRankLinear(
                        in_features=in_features,
                        out_features=out_features,
                        rank=config["rank"],
                        func_path=config["func"],
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                        bias=module.bias is not None,
                        is_compiled=config.get("compile", False)
                    )
                elif method == "dense":
                    new_layer = DenseLinear(
                        in_features=in_features,
                        out_features=out_features,
                        func_path=config["func"],
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                        bias=module.bias is not None,
                        is_compiled=config.get("compile", False)
                    )
                
                setattr(parent_module, attr_name, new_layer)

    print(f"Finished replacing nn.Linear layers in {model} with {method} layers ...")