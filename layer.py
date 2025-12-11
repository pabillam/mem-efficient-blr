import os
import importlib
import torch
import pickle
from utils import load_yaml, merge_dicts
from utils import VerificationStatus, do_verify
from triton.testing import do_bench
from tabulate import tabulate

class Layer:
    def __init__(self, config_path):
        if not config_path:
            raise ValueError("A YAML configuration file path must be provided")

        layer_config = load_yaml(config_path)
        default_config = load_yaml("./configs/config.yaml")
        config = default_config | layer_config

        self.network_name = config["network_name"]
        self.layer_name = config["layer_name"]
        self.num_seq = config["num_seq"]
        self.num_batches = config["num_batches"]
        self.in_f = config["in_f"]
        self.out_f = config["out_f"]

        self.blast = config["blast"]
        self.monarch = config.get("monarch", None)
        self.low_rank = config["low_rank"]

        def _get_func_from_str(func_path):
            module_name, func_name = func_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if callable(attr):
                    globals()[attr_name] = attr

            return getattr(module, func_name)

        def _compile_func(func_path, compile=False):
            func = _get_func_from_str(func_path)
            return torch.compile(func) if compile else func

        if ("triton_blast_verify" in config) and ("triton_blast_benchmark" in config):
            self.triton_blast_verify = [ _get_func_from_str(func_str) for func_str in sorted(config["triton_blast_verify"]["funcs"]) ]
            self.triton_blast_verification_status = [ VerificationStatus.UNKNOWN for _ in self.triton_blast_verify ]
            self.triton_blast_benchmark = [ _get_func_from_str(func_str) for func_str in sorted(config["triton_blast_benchmark"]["funcs"]) ]
            self.triton_blast_benchmark_time = [ -1 for _ in self.triton_blast_benchmark ]
            if len(self.triton_blast_verify) != len(self.triton_blast_benchmark):
                raise ValueError("Mismatch: triton_blast_verify.funcs and triton_blast_benchmark.funcs must have the same length")
        else:
            self.triton_blast_verify = []
            self.triton_blast_verification_status = None
            self.triton_blast_benchmark = []
            self.triton_blast_benchmark_time = None

        self.torch_blast_is_compiled = config["torch_blast"]["compile"]
        self.torch_blast = _compile_func(config["torch_blast"]["funcs"], self.torch_blast_is_compiled)
        self.torch_blast_benchmark_time = -1

        if ("triton_blast_sym_quant_benchmark" in config):
            self.triton_blast_sym_quant_benchmark = [ _get_func_from_str(func_str) for func_str in sorted(config["triton_blast_sym_quant_benchmark"]["funcs"]) ]
            self.triton_blast_sym_quant_benchmark_time = [ -1 for _ in self.triton_blast_sym_quant_benchmark ]
            self.triton_blast_sym_quant_verification_status = [ VerificationStatus.NA for _ in self.triton_blast_sym_quant_benchmark ]
        else:
            self.triton_blast_sym_quant_benchmark = []
            self.triton_blast_sym_quant_benchmark_time = None

        if self.monarch is not None:
            if ("triton_monarch_verify" in config) and ("triton_monarch_benchmark" in config):
                self.triton_monarch_verify = [ _get_func_from_str(func_str) for func_str in sorted(config["triton_monarch_verify"]["funcs"]) ]
                self.triton_monarch_verification_status = [ VerificationStatus.UNKNOWN for _ in self.triton_monarch_verify ]
                self.triton_monarch_benchmark = [ _get_func_from_str(func_str) for func_str in sorted(config["triton_monarch_benchmark"]["funcs"]) ]
                self.triton_monarch_benchmark_time = [ -1 for _ in self.triton_monarch_benchmark ]
                if len(self.triton_monarch_verify) != len(self.triton_monarch_benchmark):
                    raise ValueError("Mismatch: triton_monarch_verify.funcs and triton_monarch_benchmark.funcs must have the same length")
            else:
                self.triton_monarch_verify = []
                self.triton_monarch_verification_status = None
                self.triton_monarch_benchmark = []
                self.triton_monarch_benchmark_time = None

            self.torch_monarch_is_compiled = config["torch_monarch"]["compile"]
            self.torch_monarch = _compile_func(config["torch_monarch"]["funcs"], self.torch_monarch_is_compiled)
            self.torch_monarch_benchmark_time = -1
        else:
            self.triton_monarch_verify = []
            self.triton_monarch_verification_status = None
            self.triton_monarch_benchmark = []
            self.triton_monarch_benchmark_time = None
            self.torch_monarch_is_compiled = False
            self.torch_monarch = None
            self.torch_monarch_benchmark_time = None

        if ("triton_low_rank_verify" in config) and ("triton_low_rank_benchmark" in config):
            self.triton_low_rank_verify = [ _get_func_from_str(func_str) for func_str in sorted(config["triton_low_rank_verify"]["funcs"]) ]
            self.triton_low_rank_verification_status = [ VerificationStatus.UNKNOWN for _ in self.triton_low_rank_verify ]
            self.triton_low_rank_benchmark = [ _get_func_from_str(func_str) for func_str in sorted(config["triton_low_rank_benchmark"]["funcs"]) ]
            self.triton_low_rank_benchmark_time = [ -1 for _ in self.triton_low_rank_benchmark ]
            if len(self.triton_low_rank_verify) != len(self.triton_low_rank_benchmark):
                raise ValueError("Mismatch: triton_low_rank_verify.funcs and triton_low_rank_benchmark.funcs must have the same length")            
        else:
            self.triton_low_rank_verify = []
            self.triton_low_rank_benchmark = []
            self.triton_low_rank_verification_status = None
            self.triton_low_rank_benchmark_time = None
        self.torch_low_rank_is_compiled = config["torch_low_rank"]["compile"]
        self.torch_low_rank = _compile_func(config["torch_low_rank"]["funcs"], self.torch_low_rank_is_compiled)
        self.torch_low_rank_benchmark_time = -1

        if ("triton_dense_verify" in config) and ("triton_dense_benchmark" in config):
            self.triton_dense_verify = [ _get_func_from_str(func_str) for func_str in sorted(config["triton_dense_verify"]["funcs"]) ]
            self.triton_dense_verification_status = [ VerificationStatus.UNKNOWN for _ in self.triton_dense_verify ]
            self.triton_dense_benchmark = [ _get_func_from_str(func_str) for func_str in sorted(config["triton_dense_benchmark"]["funcs"]) ]
            self.triton_dense_benchmark_time = [ -1 for _ in self.triton_dense_benchmark ]
            if len(self.triton_dense_verify) != len(self.triton_dense_benchmark):
                raise ValueError("Mismatch: triton_dense_verify.funcs and triton_dense_benchmark.funcs must have the same length")  
        else:
            self.triton_dense_verify = []
            self.triton_dense_benchmark = []
            self.triton_dense_verification_status = None
            self.triton_dense_benchmark_time = None

        self.torch_dense_is_compiled = config["torch_dense"]["compile"]
        self.torch_dense = _compile_func(config["torch_dense"]["funcs"], self.torch_dense_is_compiled)
        self.torch_dense_benchmark_time = -1
        
        all_funcs = (
            self.triton_blast_benchmark +
            self.triton_blast_sym_quant_benchmark +
            (self.triton_monarch_benchmark if self.monarch is not None else []) +
            self.triton_low_rank_benchmark +
            self.triton_dense_benchmark +
            [f for f in [self.torch_blast, self.torch_monarch, self.torch_low_rank, self.torch_dense] if f is not None]
        )
        self.function_names = [func.__name__ for func in all_funcs]

    def __repr__(self):
        def get_function_names(function_list):
            if function_list is not None:
                return [func.__name__ for func in function_list]
            else:
                return []

        monarch_config_str = (
            f"B={self.monarch['b']}, Rank={self.monarch['rank']}"
            if self.monarch is not None else "N/A"
        )

        return (
            f"\n{'='*50}\n"
            f"Layer Configuration\n"
            f"{'-'*50}\n"
            f"Network Name    : {self.network_name}\n"
            f"Layer Name      : {self.layer_name}\n"
            f"Input Features  : {self.in_f}\n"
            f"Output Features : {self.out_f}\n"
            f"Num Sequence    : {self.num_seq}\n"
            f"\n"
            f"BLAST Config    : B={self.blast['b']}, Rank={self.blast['rank']}\n"
            f"Monarch Config  : {monarch_config_str}\n"
            f"Low-Rank Config : Rank={self.low_rank['rank']}\n"
            f"\n"
            f"{'='*50}\n"
            f"Microbenchmark Details\n"
            f"{'-'*50}\n"
            f"Triton Monarch Verify               : {get_function_names(self.triton_monarch_verify)}\n"
            f"Triton Monarch Benchmark            : {get_function_names(self.triton_monarch_benchmark)}\n"
            f"Triton Blast Verify                 : {get_function_names(self.triton_blast_verify)}\n"
            f"Triton Blast Benchmark              : {get_function_names(self.triton_blast_benchmark)}\n"
            f"Triton Blast Sym Quant Benchmark    : {get_function_names(self.triton_blast_sym_quant_benchmark)}\n"
            f"Triton Low-Rank Verify              : {get_function_names(self.triton_low_rank_verify)}\n"
            f"Triton Low-Rank Benchmark           : {get_function_names(self.triton_low_rank_benchmark)}\n"
            f"Triton Dense Verify                 : {get_function_names(self.triton_dense_verify)}\n"
            f"Triton Dense Benchmark              : {get_function_names(self.triton_dense_benchmark)}\n"
            f"\n"
            f"Torch BLAST    : Name={self.torch_blast.__name__} Compiled={self.torch_blast_is_compiled}\n"
            f"Torch Monarch  : Name={(self.torch_monarch.__name__ if self.torch_monarch is not None else 'N/A')} "
            f"Compiled={(self.torch_monarch_is_compiled if self.torch_monarch is not None else 'N/A')}\n"
            f"Torch Low-Rank : Name={self.torch_low_rank.__name__} Compiled={self.torch_low_rank_is_compiled}\n"
            f"Torch Dense    : Name={self.torch_dense.__name__} Compiled={self.torch_dense_is_compiled}\n"
            f"\n{'='*50}\n"
        )
    
    def verify(self, method):
        device = torch.device("cuda:0")
        dtype = torch.float32

        x = torch.randn((self.num_batches, self.num_seq, self.in_f), device=device, dtype=dtype)
        if (method == "blast" or method == "all") and self.triton_blast_verify:
            torch_U = torch.randn((self.blast["b"], self.out_f // self.blast["b"], self.blast["rank"]), device=device, dtype=dtype)
            torch_V = torch.randn((self.blast["b"], self.blast["rank"], self.in_f // self.blast["b"]), device=device, dtype=dtype)
            torch_S = torch.randn((self.blast["b"], self.blast["b"], self.blast["rank"]), device=device, dtype=dtype)

            triton_U = torch_U.transpose(1, 2).contiguous()
            triton_V = torch_V.transpose(1, 2).contiguous()
            triton_S = torch_S.transpose(0, 1).contiguous()
            
            triton_V_bmm = triton_V
            triton_S_bmm = torch_S.permute(2, 0, 1).contiguous()
            triton_U_bmm = torch_U

            torch_out = self.torch_blast(x, torch_U, torch_V, torch_S)
            for i in range(0, len(self.triton_blast_verify)):
                print(f"Verifying {self.triton_blast_verify[i].__name__} ...")
                try:
                    if self.triton_blast_verify[i].__name__.startswith("triton_blast_bmm"):
                        triton_out = self.triton_blast_verify[i](x, triton_U_bmm, triton_V_bmm, triton_S_bmm)
                    else:
                        triton_out = self.triton_blast_verify[i](x, triton_U, triton_V, triton_S)
                except Exception as e:
                    print(f"An error occured while executing {self.triton_blast_verify[i].__name__}: {e}")
                    self.triton_blast_verification_status[i] = VerificationStatus.FAIL
                    continue
                self.triton_blast_verification_status[i] = do_verify(triton_out[0], torch_out[0])

        if (method == "monarch" or method == "all") and self.monarch is not None and self.triton_monarch_verify:
            torch_w1_bfly_t = torch.randn((self.monarch["b"], self.monarch["rank"], self.in_f // self.monarch["b"]), device=device, dtype=dtype)
            torch_w2_bfly_t = torch.randn((self.monarch["b"], self.out_f // self.monarch["b"], self.monarch["rank"]), device=device, dtype=dtype)

            triton_w1_bfly = torch_w1_bfly_t.transpose(-1, -2).contiguous()
            triton_w1_bfly = triton_w1_bfly.view(self.monarch["b"], self.in_f // self.monarch["b"], self.monarch["rank"] // self.monarch["b"], self.monarch["b"])
            triton_w1_bfly = triton_w1_bfly.permute(0, 1, 3, 2)
            triton_w1_bfly = triton_w1_bfly.reshape(self.monarch["b"], self.in_f // self.monarch["b"], self.monarch["rank"]).contiguous()

            triton_w2_bfly = torch_w2_bfly_t.transpose(-1, -2).contiguous()

            torch_out = self.torch_monarch(x, torch_w1_bfly_t, torch_w2_bfly_t)
            for i in range(0, len(self.triton_monarch_verify)):
                print(f"Verifying {self.triton_monarch_verify[i].__name__} ...")
                try:
                    if self.triton_monarch_verify[i].__name__.startswith("triton_monarch_right_left"):
                        triton_out = self.triton_monarch_verify[i](x, triton_w1_bfly, triton_w2_bfly)
                        self.triton_monarch_verification_status[i] = do_verify(triton_out[0], torch_out[0])
                    elif self.triton_monarch_verify[i].__name__.startswith("triton_monarch_right_ideal"):
                        triton_out = self.triton_monarch_verify[i](x, triton_w1_bfly, torch_w2_bfly_t)
                        self.triton_monarch_verification_status[i] = do_verify(triton_out[1], torch_out[2])
                    elif self.triton_monarch_verify[i].__name__.startswith("triton_monarch_right"):
                        triton_out = self.triton_monarch_verify[i](x, triton_w1_bfly, torch_w2_bfly_t)
                        self.triton_monarch_verification_status[i] = do_verify(triton_out[0], torch_out[0])
                except Exception as e:
                    print(f"An error occured while executing {self.triton_monarch_verify[i].__name__}: {e}")
                    self.triton_monarch_verification_status[i] = VerificationStatus.FAIL
                    continue
        
        if (method == "low_rank" or method == "all") and self.triton_low_rank_verify:
            torch_Vt = torch.randn((self.low_rank["rank"], self.in_f), device=device, dtype=dtype)
            torch_Ut = torch.randn((self.out_f, self.low_rank["rank"]), device=device, dtype=dtype)

            triton_V = torch_Vt.transpose(0, 1).contiguous()
            triton_U = torch_Ut.transpose(0, 1).contiguous()
                
            torch_out = self.torch_low_rank(x, torch_Vt, torch_Ut)
            for i in range(0, len(self.triton_low_rank_verify)):
                print(f"Verifying {self.triton_low_rank_verify[i].__name__} ...")
                try:
                    triton_out = self.triton_low_rank_verify[i](x, triton_V, triton_U)
                except Exception as e:
                    print(f"An error occured while executing {self.triton_low_rank_verify[i].__name__}: {e}")
                    self.triton_low_rank_verification_status[i] = VerificationStatus.FAIL
                    continue
                self.triton_low_rank_verification_status[i] = do_verify(triton_out[0], torch_out)

        if (method == "dense" or method == "all") and self.triton_dense_verify:
            torch_Wt = torch.randn((self.out_f, self.in_f), device=device, dtype=dtype)

            triton_W = torch_Wt.transpose(0, 1).contiguous()
                
            torch_out = self.torch_dense(x, torch_Wt)
            for i in range(0, len(self.triton_dense_verify)):
                print(f"Verifying {self.triton_dense_verify[i].__name__} ...")
                try:
                    triton_out = self.triton_dense_verify[i](x, triton_W)
                except Exception as e:
                    print(f"An error occured while executing {self.triton_dense_verify[i].__name__}: {e}")
                    self.triton_dense_verification_status[i] = VerificationStatus.FAIL
                    continue
                self.triton_dense_verification_status[i] = do_verify(triton_out, torch_out)

    def benchmark(self, method):
        device = torch.device("cuda:0")
        dtype = torch.bfloat16

        x = torch.randn((self.num_batches, self.num_seq, self.in_f), device=device, dtype=dtype)
        if method == "blast" or method == "all":
            torch_U = torch.randn((self.blast["b"], self.out_f // self.blast["b"], self.blast["rank"]), device=device, dtype=dtype)
            torch_V = torch.randn((self.blast["b"], self.blast["rank"], self.in_f // self.blast["b"]), device=device, dtype=dtype)
            torch_S = torch.randn((self.blast["b"], self.blast["b"], self.blast["rank"]), device=device, dtype=dtype)

            triton_U = torch_U.transpose(1, 2).contiguous()
            triton_V = torch_V.transpose(1, 2).contiguous()
            triton_S = torch_S.transpose(0, 1).contiguous()
            
            triton_V_bmm = triton_V
            triton_S_bmm = torch_S.permute(2, 1, 0).contiguous()
            triton_U_bmm = torch_U

            base_time = do_bench(lambda: self.torch_blast(x, torch_U, torch_V, torch_S))
            self.torch_blast_benchmark_time = base_time

            for i in range(0, len(self.triton_blast_benchmark)):
                try:
                    print(f"Benchmarking {self.triton_blast_benchmark[i].__name__} ...")
                    if self.triton_blast_benchmark[i].__name__.startswith("triton_blast_bmm"):
                        triton_time = do_bench(lambda: self.triton_blast_benchmark[i](x, triton_U_bmm, triton_V_bmm, triton_S_bmm))
                    else:    
                        triton_time = do_bench(lambda: self.triton_blast_benchmark[i](x, triton_U, triton_V, triton_S))
                    self.triton_blast_benchmark_time[i] = triton_time

                    if self.triton_blast_benchmark[i].__name__.startswith("triton_blast_bmm"):
                        for suffix in ['xv', 'sxv', 'usxv']:
                            get_config = globals()[f"get_triton_blast_bmm_{suffix}_fp16_config"]
                            config = get_config()
                            output_dir = f"output/{self.network_name}/{self.layer_name}"
                            os.makedirs(output_dir, exist_ok=True)
                            output_path = os.path.join(output_dir, f"triton_blast_bmm_{suffix}_fp16_best_config.pkl")
                            with open(output_path, "wb") as f:
                                pickle.dump(config, f)
                    else:
                        get_config = globals()[f"get_{self.triton_blast_benchmark[i].__name__}_config"]
                        config = get_config()
                        output_dir = f"output/{self.network_name}/{self.layer_name}"
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, f"{self.triton_blast_benchmark[i].__name__}_best_config.pkl")
                        with open(output_path, "wb") as f:
                            pickle.dump(config, f)
                except Exception as e:
                    print(f"An error occured while executing {self.triton_blast_benchmark[i].__name__}: {e}")
                    continue
        
            for i in range(0, len(self.triton_blast_sym_quant_benchmark)):
                try:
                    print(f"Benchmarking {self.triton_blast_sym_quant_benchmark[i].__name__} ...")
                    self.triton_blast_sym_quant_benchmark_time[i] = do_bench(lambda: self.triton_blast_sym_quant_benchmark[i](x, triton_U_bmm, triton_V_bmm, triton_S_bmm))
                    for suffix in ['xv', 'sxv', 'usxv']:
                        for quant in ['int8']:
                            get_config = globals()[f"get_triton_blast_bmm_{suffix}_{quant}_fp16_config"]
                            config = get_config()
                            output_dir = f"output/{self.network_name}/{self.layer_name}"
                            os.makedirs(output_dir, exist_ok=True)
                            output_path = os.path.join(output_dir, f"triton_blast_bmm_{suffix}_{quant}_fp16_best_config.pkl")
                            with open(output_path, "wb") as f:
                                pickle.dump(config, f)
                except Exception as e:
                    print(f"An error occured while executing {self.triton_blast_sym_quant_benchmark[i].__name__}: {e}")
                    continue

        if (method == "monarch" or method == "all") and self.monarch is not None and self.triton_monarch_benchmark:
            torch_w1_bfly_t = torch.randn((self.monarch["b"], self.monarch["rank"], self.in_f // self.monarch["b"]), device=device, dtype=dtype)
            torch_w2_bfly_t = torch.randn((self.monarch["b"], self.out_f // self.monarch["b"], self.monarch["rank"]), device=device, dtype=dtype)

            triton_w1_bfly = torch_w1_bfly_t.transpose(-1, -2).contiguous()
            triton_w1_bfly = triton_w1_bfly.view(self.monarch["b"], self.in_f // self.monarch["b"], self.monarch["rank"] // self.monarch["b"], self.monarch["b"])
            triton_w1_bfly = triton_w1_bfly.permute(0, 1, 3, 2)
            triton_w1_bfly = triton_w1_bfly.reshape(self.monarch["b"], self.in_f // self.monarch["b"], self.monarch["rank"]).contiguous()

            triton_w2_bfly = torch_w2_bfly_t.transpose(-1, -2).contiguous()

            base_time = do_bench(lambda: self.torch_monarch(x, torch_w1_bfly_t, torch_w2_bfly_t))
            self.torch_monarch_benchmark_time = base_time

            for i in range(0, len(self.triton_monarch_benchmark)):
                try:
                    print(f"Benchmarking {self.triton_monarch_benchmark[i].__name__} ...")
                    if self.triton_monarch_benchmark[i].__name__.startswith("triton_monarch_right_left"):
                        triton_time = do_bench(lambda: self.triton_monarch_benchmark[i](x, triton_w1_bfly, triton_w2_bfly))
                    else:
                        triton_time = do_bench(lambda: self.triton_monarch_benchmark[i](x, triton_w1_bfly, torch_w2_bfly_t))
                    self.triton_monarch_benchmark_time[i] = triton_time

                    get_config = globals()[f"get_{self.triton_monarch_benchmark[i].__name__}_config"]
                    config = get_config()
                    output_dir = f"output/{self.network_name}/{self.layer_name}"
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"{self.triton_monarch_benchmark[i].__name__}_best_config.pkl")
                    with open(output_path, "wb") as f:
                        pickle.dump(config, f)

                except Exception as e:
                    print(f"An error occured while executing {self.triton_monarch_benchmark[i].__name__}: {e}")
                    continue

        if method == "low_rank" or method == "all":
            torch_Vt = torch.randn((self.low_rank["rank"], self.in_f), device=device, dtype=dtype)
            torch_Ut = torch.randn((self.out_f, self.low_rank["rank"]), device=device, dtype=dtype)

            triton_V = torch_Vt.transpose(0, 1).contiguous()
            triton_U = torch_Ut.transpose(0, 1).contiguous()
                
            base_time = do_bench(lambda: self.torch_low_rank(x, torch_Vt, torch_Ut))
            self.torch_low_rank_benchmark_time = base_time

            for i in range(0, len(self.triton_low_rank_benchmark)):
                print(f"Benchmarking {self.triton_low_rank_benchmark[i].__name__} ...")
                try:
                    triton_time = do_bench(lambda: self.triton_low_rank_benchmark[i](x, triton_V, triton_U))
                    self.triton_low_rank_benchmark_time[i] = triton_time

                    get_config = globals()[f"get_{self.triton_low_rank_benchmark[i].__name__}_config"]
                    config = get_config()
                    output_dir = f"output/{self.network_name}/{self.layer_name}"
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"{self.triton_low_rank_benchmark[i].__name__}_best_config.pkl")
                    with open(output_path, "wb") as f:
                        pickle.dump(config, f)

                except Exception as e:
                    print(f"An error occured while executing {self.triton_low_rank_benchmark[i].__name__}: {e}")
                    continue

        if method == "dense" or method == "all":
            torch_Wt = torch.randn((self.out_f, self.in_f), device=device, dtype=dtype)

            triton_W = torch_Wt.transpose(0, 1).contiguous()
                
            base_time = do_bench(lambda: self.torch_dense(x, torch_Wt))
            self.torch_dense_benchmark_time = base_time

            for i in range(0, len(self.triton_dense_benchmark)):
                print(f"Benchmarking {self.triton_dense_benchmark[i].__name__} ...")
                try:
                    triton_time = do_bench(lambda: self.triton_dense_benchmark[i](x, triton_W))
                    self.triton_dense_benchmark_time[i] = triton_time

                    get_config = globals()[f"get_{self.triton_dense_benchmark[i].__name__}_config"]
                    config = get_config()
                    output_dir = f"output/{self.network_name}/{self.layer_name}"
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"{self.triton_dense_benchmark[i].__name__}_best_config.pkl")
                    with open(output_path, "wb") as f:
                        pickle.dump(config, f)

                except Exception as e:
                    print(f"An error occured while executing {self.triton_dense_benchmark[i].__name__}: {e}")
                    continue

    def summary(self):
        torch_results = [
            (("torch.compile(" + self.torch_blast.__name__ + ")") if self.torch_blast_is_compiled else self.torch_blast.__name__, 
            self.torch_blast_benchmark_time, "N/A"),
        ]

        if self.torch_monarch is not None:
            torch_results.append(
                (("torch.compile(" + self.torch_monarch.__name__ + ")") if self.torch_monarch_is_compiled else self.torch_monarch.__name__, 
                self.torch_monarch_benchmark_time, "N/A")
            )

        torch_results.extend([
            (("torch.compile(" + self.torch_low_rank.__name__ + ")") if self.torch_low_rank_is_compiled else self.torch_low_rank.__name__, 
            self.torch_low_rank_benchmark_time, "N/A"),
            (("torch.compile(" + self.torch_dense.__name__ + ")") if self.torch_dense_is_compiled else self.torch_dense.__name__, 
            self.torch_dense_benchmark_time, "N/A")
        ])

        table = torch_results

        if self.triton_blast_benchmark:
            blast_results = sorted(
                zip(self.triton_blast_benchmark, self.triton_blast_benchmark_time, self.triton_blast_verification_status),
                key=lambda x: x[1] if x[1] != -1 else float('inf')
            )
            table.append([None, None, None])
            table.extend([[func.__name__, time, status.name] for func, time, status in blast_results])

        if self.triton_blast_sym_quant_benchmark:
            blast_sym_quant_results = sorted(
                zip(self.triton_blast_sym_quant_benchmark, self.triton_blast_sym_quant_benchmark_time, self.triton_blast_sym_quant_verification_status),
                key=lambda x: x[1] if x[1] != -1 else float('inf')
            )
            table.append([None, None, None])
            table.extend([[func.__name__, time, status.name] for func, time, status in blast_sym_quant_results])

        if self.monarch is not None and self.triton_monarch_benchmark:
            monarch_results = sorted(
                zip(self.triton_monarch_benchmark, self.triton_monarch_benchmark_time, self.triton_monarch_verification_status),
                key=lambda x: x[1] if x[1] != -1 else float('inf')
            )
            table.append([None, None, None])
            table.extend([[func.__name__, time, status.name] for func, time, status in monarch_results])

        if self.triton_low_rank_benchmark:
            low_rank_results = sorted(
                zip(self.triton_low_rank_benchmark, self.triton_low_rank_benchmark_time, self.triton_low_rank_verification_status),
                key=lambda x: x[1] if x[1] != -1 else float('inf')
            )
            table.append([None, None, None])
            table.extend([[func.__name__, time, status.name] for func, time, status in low_rank_results])

        if self.triton_dense_benchmark:
            dense_results = sorted(
                zip(self.triton_dense_benchmark, self.triton_dense_benchmark_time, self.triton_dense_verification_status),
                key=lambda x: x[1] if x[1] != -1 else float('inf')
            )
            table.append([None, None, None])
            table.extend([[func.__name__, time, status.name] for func, time, status in dense_results])

        headers = ["Function", "Benchmark Time (ms)", "Verification Status"]
        table_str = tabulate(table, headers=headers, tablefmt="grid", missingval="—")
        print(table_str)

        output_dir = f"output/{self.network_name}/{self.layer_name}"
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "summary.txt")
        with open(output_path, "w") as f:
            f.write(table_str)

if __name__ == '__main__':
    network = "dit_xl"
    layer = "adaln"
    method = "all"

    layer_config_path = f"./configs/layers/{network}.{layer}.yaml"
    l = Layer(layer_config_path)
    print(l)

    l.benchmark(method)
    l.verify(method)
    l.summary()
