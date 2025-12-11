import os
import torch
from utils import load_yaml
import importlib
from tabulate import tabulate
import subprocess

class Network():
    def __init__(self, config_path):
        if not config_path:
            raise ValueError("A YAML configuration file path must be provided")
        network_config = load_yaml(config_path)

        self.name = network_config["name"]
        module = importlib.import_module(f"models.{self.name}")
        self.benchmark_func = getattr(module, "benchmark")
        self.benchmark_results = {}

        self.num_seq = network_config["num_seq"]
        self.layer_configs = {}
        for method in ["blast", "blast_sym_quant", "monarch", "low_rank", "dense"]:
            if method not in network_config:
                if method == "monarch" or method == "blast_sym_quant":
                    continue
                else:
                    raise KeyError(f"Method '{method}' missing in network configuration")

            self.layer_configs[method] = {}
            self.layer_configs[method]["triton"] = {}
            self.layer_configs[method]["torch"] = {}

            for layer, details in network_config[method].items():
                if not isinstance(details, dict):
                    continue

                if "triton" in details:
                    self.layer_configs[method]["triton"][layer] = {
                        "rank": details.get("rank"),
                        "b": details.get("b"),
                        "func": details["triton"].get("func"),
                    }

                if "torch" in details:
                    self.layer_configs[method]["torch"][layer] = {
                        "rank": details.get("rank"),
                        "b": details.get("b"),
                        "func": details["torch"].get("func"),
                        "compile": details["torch"].get("compile", False),
                    }

    def benchmark(self, method_filter=["blast", "blast_sym_quant", "monarch", "low_rank", "dense"]):
        device = torch.device("cuda")
        dtype = torch.bfloat16
        num_iters = 100

        for method in method_filter:
            for framework in self.layer_configs[method]:
                if self.layer_configs[method][framework]:
                    layer_configs = self.layer_configs[method][framework]
                    benchmark_time = self.benchmark_func(method, framework, layer_configs, self.num_seq, num_iters, device, dtype)
                    self.benchmark_results.setdefault(method, {}).setdefault(framework, {})["time"] = benchmark_time

    def summary(self):
        table_data = []
        for method, frameworks in self.benchmark_results.items():
            if method not in self.layer_configs:
                continue

            for framework, results in frameworks.items():
                table_data.append([method, framework, results["time"]])
        
        table_str = tabulate(table_data, headers=["Method", "Framework", "Benchmark Time (ms)"], tablefmt="grid")
        print(table_str)
        output_dir = f"output/{self.name}"
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"end2end_summary_{self.num_seq}.txt")
        with open(output_path, "w") as f:
            f.write(table_str)

