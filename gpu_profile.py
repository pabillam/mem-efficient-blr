import os
import torch
import argparse
import sys
import pickle
import torch.utils.benchmark as benchmark
from triton_funcs.blast_kernels import *
from triton_funcs.monarch_kernels import *
from triton_funcs.dense_kernels import *
from triton_funcs.low_rank_kernels import *
from triton_funcs.blast_sym_quant_kernels import *
from torch_funcs.blast_funcs import *
from torch_funcs.monarch_funcs import *
from torch_funcs.dense_funcs import *
from torch_funcs.low_rank_funcs import *

def main(args):
    function = args.function
    num_iters = args.num_iters
    num_warmup_iters = args.num_warmup_iters
    num_batches = args.num_batches
    num_seq = args.num_seq
    in_f = args.in_f
    out_f = args.out_f
    b = args.b
    rank = args.rank
    mode = args.mode
    config_dir = args.config_dir

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    x = torch.randn((num_batches, num_seq, in_f), device=device, dtype=dtype)

    if "blast" in function.__name__:
        if "torch" in function.__name__:
            U = torch.randn((b, out_f // b, rank), device=device, dtype=dtype)
            V = torch.randn((b, rank, in_f // b), device=device, dtype=dtype)
            S = torch.randn((b, b, rank), device=device, dtype=dtype)
            best_config = None
        elif "triton" in function.__name__:
            if "bmm" in function.__name__:
                U = torch.randn((b, out_f // b, rank), device=device, dtype=dtype)
                V = torch.randn((b, in_f // b, rank), device=device, dtype=dtype)
                S = torch.randn((rank, b, b), device=device, dtype=dtype)
                if "int8" in function.__name__:
                    dtype_suffix = '_int8'
                else:
                    dtype_suffix = ''
                best_config = []
                for suffix in ['xv', 'sxv', 'usxv']:
                    best_config_path = os.path.join(config_dir, f"triton_blast_bmm_{suffix}{dtype_suffix}_fp16_best_config.pkl")
                    if not os.path.exists(best_config_path):
                        print(f"Skipping {function.__name__} as best_config file not found: {best_config_path}")
                        return
                    with open(best_config_path, 'rb') as f:
                        best_config_temp = pickle.load(f)
                    best_config.append(best_config_temp)
                best_config = tuple(best_config)
            else:
                U = torch.randn((b, rank, out_f // b), device=device, dtype=dtype)
                V = torch.randn((b, in_f // b, rank), device=device, dtype=dtype)
                S = torch.randn((b, b, rank), device=device, dtype=dtype)
                best_config_path = os.path.join(config_dir, f"{function.__name__}_best_config.pkl")
                if not os.path.exists(best_config_path):
                    print(f"Skipping {function.__name__} as best_config file not found: {best_config_path}")
                    return
                with open(best_config_path, 'rb') as f:
                    best_config = pickle.load(f)

        if mode == "nsys":
            for i in range(num_iters):
                if i == num_warmup_iters: torch.cuda.cudart().cudaProfilerStart()
                if i >= num_warmup_iters: torch.cuda.nvtx.range_push(f"{function.__name__} Iteration {i}")
                function(x, U, V, S, best_config)
                if i >= num_warmup_iters: torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
        elif mode == "ncu":
            t = benchmark.Timer(
                stmt='m(x, U, V, S, best_config)',
                globals={'m': function, 'x': x, 'U': U, 'V': V, 'S': S, 'best_config': best_config}
            )
            temp = t.timeit(num_iters)

    if "monarch" in function.__name__:
        if "torch" in function.__name__:
            w1_bfly = torch.randn((b, rank, in_f // b), device=device, dtype=dtype)
            w2_bfly = torch.randn((b, out_f // b, rank), device=device, dtype=dtype)
            best_config = None
        elif "triton" in function.__name__:                
            w1_bfly = torch.randn((b, in_f // b, rank), device=device, dtype=dtype)
            if "left" in function.__name__:
                w2_bfly = torch.randn((b, rank, out_f // b, ), device=device, dtype=dtype)
            elif "right" in function.__name__:
                w2_bfly = torch.randn((b, out_f // b, rank), device=device, dtype=dtype)
            best_config_path = os.path.join(config_dir, f"{function.__name__}_best_config.pkl")
            if not os.path.exists(best_config_path):
                print(f"Skipping {function.__name__} as best_config file not found: {best_config_path}")
                return
            with open(best_config_path, 'rb') as f:
                best_config = pickle.load(f)

        if mode == "nsys":
            for i in range(num_iters):
                if i == num_warmup_iters: torch.cuda.cudart().cudaProfilerStart()
                if i >= num_warmup_iters: torch.cuda.nvtx.range_push(f"{function.__name__} Iteration {i}")
                function(x, w1_bfly, w2_bfly, best_config)
                if i >= num_warmup_iters: torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
        elif mode == "ncu":
            t = benchmark.Timer(
                stmt='m(x, w1_bfly, w2_bfly, best_config)',
                globals={'m': function, 'x': x, 'w1_bfly': w1_bfly, 'w2_bfly': w2_bfly, 'best_config': best_config}
            )
            temp = t.timeit(num_iters)

    if "low_rank" in function.__name__:
        if "torch" in function.__name__:
            V = torch.randn((rank, in_f), device=device, dtype=dtype)
            U = torch.randn((out_f, rank), device=device, dtype=dtype)
            best_config = None
        elif "triton" in function.__name__:
            V = torch.randn((in_f, rank), device=device, dtype=dtype)
            U = torch.randn((rank, out_f), device=device, dtype=dtype)
            best_config_path = os.path.join(config_dir, f"{function.__name__}_best_config.pkl")
            if not os.path.exists(best_config_path):
                print(f"Skipping {function.__name__} as best_config file not found: {best_config_path}")
                return
            with open(best_config_path, 'rb') as f:
                best_config = pickle.load(f)
                
        if mode == "nsys":
            for i in range(num_iters):
                if i == num_warmup_iters: torch.cuda.cudart().cudaProfilerStart()
                if i >= num_warmup_iters: torch.cuda.nvtx.range_push(f"{function.__name__} Iteration {i}")
                function(x, V, U, best_config)
                if i >= num_warmup_iters: torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
        elif mode == "ncu":
            t = benchmark.Timer(
                stmt='m(x, V, U, best_config)',
                globals={'m': function, 'x': x, 'V': V, 'U': U, 'best_config': best_config}
            )
            temp = t.timeit(num_iters)
    
    if "dense" in function.__name__:
        if "torch" in function.__name__:
            W = torch.randn((out_f, in_f), device=device, dtype=dtype)
            best_config = None
        elif "triton" in function.__name__:
            W = torch.randn((in_f, out_f), device=device, dtype=dtype)
            best_config_path = os.path.join(config_dir, f"{function.__name__}_best_config.pkl")
            if not os.path.exists(best_config_path):
                print(f"Skipping {function.__name__} as best_config file not found: {best_config_path}")
                return
            with open(best_config_path, 'rb') as f:
                best_config = pickle.load(f)

        if mode == "nsys":
            for i in range(num_iters):
                if i == num_warmup_iters: torch.cuda.cudart().cudaProfilerStart()
                if i >= num_warmup_iters: torch.cuda.nvtx.range_push(f"{function.__name__} Iteration {i}")
                function(x, W, best_config)
                if i >= num_warmup_iters: torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
        elif mode == "ncu":
            t = benchmark.Timer(
                stmt='m(x, W, best_config)',
                globals={'m': function, 'x': x, 'W': W, 'best_config': best_config}
            )
            temp = t.timeit(num_iters)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--function_name",
                        type=str,
                        required=True,
                        help="Function to profile")
    parser.add_argument("--num_batches",
                        type=int,
                        required=True,
                        help="Batch size of tensor")
    parser.add_argument("--num_seq",
                        type=int,
                        required=True,
                        help="Number of input rows")
    parser.add_argument("--in_f",
                        type=int,
                        required=True,
                        help="Number of input columns")
    parser.add_argument("--out_f",
                        type=int,
                        required=True,
                        help="Number of output columns")
    parser.add_argument("--b",
                        type=int,
                        default=1,
                        help="Number of blocks")
    parser.add_argument("--rank",
                        type=int,
                        default=0,
                        help="Rank of matrix")
    parser.add_argument("--mode",
                        type=str,
                        required=True,
                        choices=["ncu", "nsys"],
                        help="Indicates whether the script is intended to run with ncu or nsys")
    parser.add_argument("--compile",
                        action="store_true",
                        help="Compiles function")
    parser.add_argument("--config_dir",
                        type=str,
                        required=True,
                        help="Path to Tritonkernel config")


    args = parser.parse_args()
    _, args.function_name = args.function_name.rsplit(".", 1)
    args.function = globals()[args.function_name]
    if args.compile:
        args.function = torch.compile(args.function)

    if args.mode == "ncu":
        args.num_iters = 1
        args.num_warmup_iters = 1
    elif args.mode == "nsys":
        args.num_iters = 20
        args.num_warmup_iters = 10

    main(args)