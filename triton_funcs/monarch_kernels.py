import torch
import triton
import triton.language as tl
import numpy as np
from utils import next_power_of_2

""" Autotune Configurations for Triton Monarch Kernels """
def _get_triton_monarch_right_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_R': 16,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_R': 64,  'BLOCK_SIZE_P': 128, 'GROUP_SIZE_N': 8},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_R': 64,  'BLOCK_SIZE_P': 64,  'GROUP_SIZE_N': 8},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_R': 64,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8},  num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_R': 16,  'BLOCK_SIZE_P': 64,  'GROUP_SIZE_N': 8},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_R': 16,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_R': 16,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 128, 'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 64,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 128, 'GROUP_SIZE_N': 8},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 64,  'GROUP_SIZE_N': 8},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8},  num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 64,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 64,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8},  num_stages=5, num_warps=2)
    ]

def _get_triton_monarch_left_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_Q': 16,  'BLOCK_SIZE_R': 32,  'GROUP_SIZE_N': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_Q': 64,  'BLOCK_SIZE_R': 128, 'GROUP_SIZE_N': 8},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_Q': 64,  'BLOCK_SIZE_R': 64,  'GROUP_SIZE_N': 8},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_Q': 64,  'BLOCK_SIZE_R': 32,  'GROUP_SIZE_N': 8},  num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_Q': 16,  'BLOCK_SIZE_R': 64,  'GROUP_SIZE_N': 8},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_Q': 16,  'BLOCK_SIZE_R': 32,  'GROUP_SIZE_N': 8},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_Q': 16,  'BLOCK_SIZE_R': 32,  'GROUP_SIZE_N': 8},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_Q': 128, 'BLOCK_SIZE_R': 32,  'GROUP_SIZE_N': 8},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_Q': 64,  'BLOCK_SIZE_R': 32,  'GROUP_SIZE_N': 8},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_Q': 32,  'BLOCK_SIZE_R': 128, 'GROUP_SIZE_N': 8},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_Q': 32,  'BLOCK_SIZE_R': 64,  'GROUP_SIZE_N': 8},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_Q': 32,  'BLOCK_SIZE_R': 32,  'GROUP_SIZE_N': 8},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_Q': 32,  'BLOCK_SIZE_R': 32,  'GROUP_SIZE_N': 8},  num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_Q': 32,  'BLOCK_SIZE_R': 32,  'GROUP_SIZE_N': 8},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_Q': 64,  'BLOCK_SIZE_R': 32,  'GROUP_SIZE_N': 8},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_Q': 64,  'BLOCK_SIZE_R': 32,  'GROUP_SIZE_N': 8},  num_stages=5, num_warps=2)
    ]

""" Triton Monarch Kernels """
#-----------------------------------
@triton.jit
def _triton_monarch_right_kernel_fp32(
    x_ptr, w1_bfly_ptr, out1_ptr, pre_out1_ptr,
    N, P, B, R, PER_BLOCK_R,
    stride_xb1, stride_xn, stride_xp,
    stride_w1_bflyb1, stride_w1_bflyp, stride_w1_bflyr,
    stride_out1b2, stride_out1n, stride_out1r,
    stride_pre_out1b1, stride_pre_out1n, stride_pre_out1r,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_P: tl.constexpr, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_R: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr
    ):

    """
    Computes a batched matrix multiplication between x and w1_bfly. The resulting product is stored in both 
    pre_out1 (intermediate results) and out1_ptr (reordered with strided memory layout). Uses FP32 inputs and outputs.
    Useful for checking correctness. Assumes innermost dimension of w1_bfly is contiguous along the per-block rank.

    Parameters:
    ----------
    x_ptr : tl.tensor
        Pointer to input matrix x, shape (B, N, P)
    w1_bfly_ptr : tl.tensor
        Pointer to butterfly weight matrix w1_bfly, shape (B, P, PER_BLOCK_R * B) where R = PER_BLOCK_R * B
    out1_ptr : tl.tensor
        Pointer to output matrix out1, shape (B, N, R)
    pre_out1_ptr : tl.tensor
        Pointer to intermediate output matrix pre_out_1, shape (B, N, R)

    N, P, B, R, PER_BLOCK_R : int
        Matrix dimensions:
        - N: Number of rows in x
        - P: Shared inner dimension between x and w1_bfly
        - B: Number of blocks
        - R: Product of B and per-block rank
        - PER_BLOCK_R: Per-block rank

    stride_xb1, stride_xn, stride_xp : int
        Strides for accessing x_ptr
    stride_w1_bflyb1, stride_w1_bflyp, stride_w1_bflyr : int
        Strides for accessing w1_bfly_ptr
    stride_out1b2, stride_out1n, stride_out1r : int
        Strides for accessing out1_ptr
    stride_pre_out1b1, stride_pre_out1n, stride_pre_out1r : int
        Strides for accessing pre_out1_ptr

    BLOCK_SIZE_N, BLOCK_SIZE_P, BLOCK_SIZE_B, BLOCK_SIZE_R : tl.constexpr
        Block sizes used for tiling computations
    GROUP_SIZE_N : tl.constexpr
        Number of program IDs grouped together to improve memory locality
    """

    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N) 
    num_pid_r = tl.cdiv(R, BLOCK_SIZE_R)
    num_pid_in_group = GROUP_SIZE_N * num_pid_r
    group_id = pid // num_pid_in_group
    first_pid_n = group_id * GROUP_SIZE_N
    group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_N)
    pid_n = first_pid_n + ((pid % num_pid_in_group) % group_size_n)
    pid_r = (pid % num_pid_in_group) // group_size_n

    b1 = tl.program_id(axis=1)
    offs_xn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_r = (pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)) % R
    offs_p = tl.arange(0, BLOCK_SIZE_P)
    
    offs_out1n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_out1r = (pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)) % PER_BLOCK_R + b1 * PER_BLOCK_R
    offs_out1b2 = (pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)) // PER_BLOCK_R

    offs_pre_out1n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_pre_out1r = (pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R))

    x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_p[None, :] * stride_xp + b1 * stride_xb1)
    w1_bfly_ptrs = w1_bfly_ptr + (offs_p[:, None] * stride_w1_bflyp + offs_r[None, :] * stride_w1_bflyr + b1 * stride_w1_bflyb1)
    accumulator_in = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
    for p in range(0, tl.cdiv(P, BLOCK_SIZE_P)):
        x_mask = (offs_p[None, :] < (P - p * BLOCK_SIZE_P))
        w1_bfly_mask = (offs_p[:, None] < (P - p * BLOCK_SIZE_P))
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w1_bfly = tl.load(w1_bfly_ptrs, mask=w1_bfly_mask, other=0.0)
        accumulator_in = tl.dot(x, w1_bfly, accumulator_in, allow_tf32=False)
        x_ptrs += BLOCK_SIZE_P * stride_xp
        w1_bfly_ptrs += BLOCK_SIZE_P * stride_w1_bflyp
    
    out1 = accumulator_in
    pre_out1_ptrs = pre_out1_ptr + (stride_pre_out1n * offs_pre_out1n[:, None] + stride_pre_out1r * offs_pre_out1r[None, :] + b1 * stride_pre_out1b1)
    out1_ptrs = out1_ptr + (stride_out1n * offs_out1n[:, None] + (stride_out1r * offs_out1r[None, :] +  offs_out1b2[None, :] * stride_out1b2))
    
    pre_out1_mask = ((offs_pre_out1n[:, None] < N) & (offs_pre_out1r[None, :] < R))
    out1_mask = ((offs_out1n[:, None] < N) & (offs_out1r[None, :] < R)) & (offs_out1b2[None, :] < B)
    tl.store(pre_out1_ptrs, out1, mask=pre_out1_mask)
    tl.store(out1_ptrs, out1, mask=out1_mask)

@triton.autotune(configs=_get_triton_monarch_right_autotune_config(), key=['N', 'P', 'B', 'R'])
@triton.jit
def _triton_monarch_right_kernel_fp16(
    x_ptr, w1_bfly_ptr, out1_ptr,
    N, P, B, R, PER_BLOCK_R,
    stride_xb1, stride_xn, stride_xp,
    stride_w1_bflyb1, stride_w1_bflyp, stride_w1_bflyr,
    stride_out1b2, stride_out1n, stride_out1r,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_P: tl.constexpr, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_R: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr
    ):

    """
    Computes a batched matrix multiplication between x and w1_bfly. The resulting product is stored in out1_ptr 
    (reordered with strided memory layout). Uses FP16 inputs and outputs. Useful for performance benchmarking. 
    Assumes innermost dimension of w1_bfly is contiguous along the per-block rank. Note sub-optimal if PER_BLOCK_R % 16 != 0.

    Parameters:
    ----------
    x_ptr : tl.tensor
        Pointer to input matrix x, shape (B, N, P)
    w1_bfly_ptr : tl.tensor
        Pointer to butterfly weight matrix w1_bfly, shape (B, P, PER_BLOCK_R * B) where R = PER_BLOCK_R * B
    out1_ptr : tl.tensor
        Pointer to output matrix out1, shape (B, N, R)

    N, P, B, R, PER_BLOCK_R : int
        Matrix dimensions:
        - N: Number of rows in x
        - P: Shared inner dimension between x and w1_bfly
        - B: Number of blocks
        - R: Product of B and per-block rank
        - PER_BLOCK_R: Per-block rank

    stride_xb1, stride_xn, stride_xp : int
        Strides for accessing x_ptr
    stride_w1_bflyb1, stride_w1_bflyp, stride_w1_bflyr : int
        Strides for accessing w1_bfly_ptr
    stride_out1b2, stride_out1n, stride_out1r : int
        Strides for accessing out1_ptr
        
    BLOCK_SIZE_N, BLOCK_SIZE_P, BLOCK_SIZE_B, BLOCK_SIZE_R : tl.constexpr
        Block sizes used for tiling computations
    GROUP_SIZE_N : tl.constexpr
        Number of program IDs grouped together to improve memory locality
    """

    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N) 
    num_pid_r = tl.cdiv(R, BLOCK_SIZE_R)
    num_pid_in_group = GROUP_SIZE_N * num_pid_r
    group_id = pid // num_pid_in_group
    first_pid_n = group_id * GROUP_SIZE_N
    group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_N)
    pid_n = first_pid_n + ((pid % num_pid_in_group) % group_size_n)
    pid_r = (pid % num_pid_in_group) // group_size_n

    b1 = tl.program_id(axis=1)
    offs_xn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_r = (pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)) % R
    offs_p = tl.arange(0, BLOCK_SIZE_P)
    
    offs_out1n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_out1r = (pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)) % PER_BLOCK_R + b1 * PER_BLOCK_R
    offs_out1b2 = (pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)) // PER_BLOCK_R

    x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_p[None, :] * stride_xp + b1 * stride_xb1)
    w1_bfly_ptrs = w1_bfly_ptr + (offs_p[:, None] * stride_w1_bflyp + offs_r[None, :] * stride_w1_bflyr + b1 * stride_w1_bflyb1)
    accumulator_in = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
    for p in range(0, tl.cdiv(P, BLOCK_SIZE_P)):
        x_mask = (offs_p[None, :] < (P - p * BLOCK_SIZE_P))
        w1_bfly_mask = (offs_p[:, None] < (P - p * BLOCK_SIZE_P))
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w1_bfly = tl.load(w1_bfly_ptrs, mask=w1_bfly_mask, other=0.0)
        accumulator_in = tl.dot(x, w1_bfly, accumulator_in)
        x_ptrs += BLOCK_SIZE_P * stride_xp
        w1_bfly_ptrs += BLOCK_SIZE_P * stride_w1_bflyp
    
    out1 = accumulator_in.to(tl.bfloat16)
    out1_ptrs = out1_ptr + (stride_out1n * offs_out1n[:, None] + (stride_out1r * offs_out1r[None, :] + offs_out1b2[None, :] * stride_out1b2))
    
    out1_mask = ((offs_out1n[:, None] < N) & (offs_out1r[None, :] < R)) & (offs_out1b2[None, :] < B)
    tl.store(out1_ptrs, out1, mask=out1_mask)

@triton.jit
def _triton_monarch_right_kernel_fp16_no_autotune(
    x_ptr, w1_bfly_ptr, out1_ptr,
    N, P, B, R, PER_BLOCK_R,
    stride_xb1, stride_xn, stride_xp,
    stride_w1_bflyb1, stride_w1_bflyp, stride_w1_bflyr,
    stride_out1b2, stride_out1n, stride_out1r,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_P: tl.constexpr, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_R: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr
    ):

    """
    Computes a batched matrix multiplication between x and w1_bfly. The resulting product is stored in out1_ptr 
    (reordered with strided memory layout). Uses FP16 inputs and outputs. Useful for performance benchmarking. 
    Assumes innermost dimension of w1_bfly is contiguous along the per-block rank. Note sub-optimal if PER_BLOCK_R % 16 != 0.

    Parameters:
    ----------
    x_ptr : tl.tensor
        Pointer to input matrix x, shape (B, N, P)
    w1_bfly_ptr : tl.tensor
        Pointer to butterfly weight matrix w1_bfly, shape (B, P, PER_BLOCK_R * B) where R = PER_BLOCK_R * B
    out1_ptr : tl.tensor
        Pointer to output matrix out1, shape (B, N, R)

    N, P, B, R, PER_BLOCK_R : int
        Matrix dimensions:
        - N: Number of rows in x
        - P: Shared inner dimension between x and w1_bfly
        - B: Number of blocks
        - R: Product of B and per-block rank
        - PER_BLOCK_R: Per-block rank

    stride_xb1, stride_xn, stride_xp : int
        Strides for accessing x_ptr
    stride_w1_bflyb1, stride_w1_bflyp, stride_w1_bflyr : int
        Strides for accessing w1_bfly_ptr
    stride_out1b2, stride_out1n, stride_out1r : int
        Strides for accessing out1_ptr
        
    BLOCK_SIZE_N, BLOCK_SIZE_P, BLOCK_SIZE_B, BLOCK_SIZE_R : tl.constexpr
        Block sizes used for tiling computations
    GROUP_SIZE_N : tl.constexpr
        Number of program IDs grouped together to improve memory locality
    """

    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N) 
    num_pid_r = tl.cdiv(R, BLOCK_SIZE_R)
    num_pid_in_group = GROUP_SIZE_N * num_pid_r
    group_id = pid // num_pid_in_group
    first_pid_n = group_id * GROUP_SIZE_N
    group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_N)
    pid_n = first_pid_n + ((pid % num_pid_in_group) % group_size_n)
    pid_r = (pid % num_pid_in_group) // group_size_n

    b1 = tl.program_id(axis=1)
    offs_xn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_r = (pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)) % R
    offs_p = tl.arange(0, BLOCK_SIZE_P)
    
    offs_out1n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_out1r = (pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)) % PER_BLOCK_R  + b1 * PER_BLOCK_R
    offs_out1b2 = (pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)) // PER_BLOCK_R

    x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_p[None, :] * stride_xp + b1 * stride_xb1)
    w1_bfly_ptrs = w1_bfly_ptr + (offs_p[:, None] * stride_w1_bflyp + offs_r[None, :] * stride_w1_bflyr + b1 * stride_w1_bflyb1)
    accumulator_in = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
    for p in range(0, tl.cdiv(P, BLOCK_SIZE_P)):
        x_mask = (offs_p[None, :] < (P - p * BLOCK_SIZE_P))
        w1_bfly_mask = (offs_p[:, None] < (P - p * BLOCK_SIZE_P))
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w1_bfly = tl.load(w1_bfly_ptrs, mask=w1_bfly_mask, other=0.0)
        accumulator_in = tl.dot(x, w1_bfly, accumulator_in)
        x_ptrs += BLOCK_SIZE_P * stride_xp
        w1_bfly_ptrs += BLOCK_SIZE_P * stride_w1_bflyp
    
    out1 = accumulator_in.to(tl.bfloat16)
    out1_ptrs = out1_ptr + (stride_out1n * offs_out1n[:, None] + (stride_out1r * offs_out1r[None, :] +  offs_out1b2[None, :] * stride_out1b2))
    
    out1_mask = ((offs_out1n[:, None] < N) & (offs_out1r[None, :] < R)) & (offs_out1b2[None, :] < B)
    tl.store(out1_ptrs, out1, mask=out1_mask)

#-----------------------------------
@triton.jit
def _triton_monarch_left_kernel_fp32(
    out1_ptr, w2_bfly_ptr, out2_ptr, pre_out2_ptr,
    N, R, B, Q,
    stride_out1b2, stride_out1n, stride_out1r,
    stride_w2_bflyb2, stride_w2_bflyr, stride_w2_bflyq,
    stride_out2n, stride_out2qb2,
    stride_pre_out2b2, stride_pre_out2n, stride_pre_out2q,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_Q: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr
    ):

    """
    Computes a batched matrix multiplication between out1 and w2_bfly. The resulting product is stored in both 
    pre_out2 (intermediate results) and out2_ptr (reordered with strided memory layout). Uses FP32 inputs and outputs.
    Useful for checking correctness. Assumes innermost dimension of w2_bfly is contiguous along the per-block rank, and
    that innermost dimension of out2 is contiguous along the block (not the per-block rank)

    Parameters:
    ----------
    out1_ptr : tl.tensor
        Pointer to the input matrix out1, shape (B, N, R)
    w2_bfly_ptr : tl.tensor
        Pointer to butterfly weight matrix w2_bfly, shape (B, R, Q)
    out2_ptr : tl.tensor
        Pointer to output matrix out2, shape (N, Q * B)
    pre_out2_ptr : tl.tensor
        Pointer to an intermediate output matrix pre_out2, shape (B, N, Q)

    N, R, B, Q : int
        Matrix dimensions:
        - N: Number of rows in out1
        - R: Product of per-block rank and B
        - B: Number of blocks
        - Q: Number of output features per block

    stride_out1b2, stride_out1n, stride_out1r : int
        Strides for accessing out1_ptr
    stride_w2_bflyb2, stride_w2_bflyr, stride_w2_bflyq : int
        Strides for accessing w2_bfly_ptr
    stride_out2n, stride_out2qb2 : int
        Strides for accessing out2_ptr
    stride_pre_out2b2, stride_pre_out2n, stride_pre_out2q : int
        Strides for accessing pre_out2_ptr

    BLOCK_SIZE_N, BLOCK_SIZE_R, BLOCK_SIZE_B, BLOCK_SIZE_Q : tl.constexpr
        Block sizes used for tiling computations
    GROUP_SIZE_N : tl.constexpr
        Number of program IDs grouped together to improve memory locality
    """

    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N) 
    num_pid_q = tl.cdiv(Q, BLOCK_SIZE_Q)
    num_pid_in_group = GROUP_SIZE_N * num_pid_q
    group_id = pid // num_pid_in_group
    first_pid_n = group_id * GROUP_SIZE_N
    group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_N)
    pid_n = first_pid_n + ((pid % num_pid_in_group) % group_size_n)
    pid_q = (pid % num_pid_in_group) // group_size_n

    offs_out1n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_q = (pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) % Q
    offs_r = tl.arange(0, BLOCK_SIZE_R)
    
    offs_out2n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_out2qb2 = (pid_q * BLOCK_SIZE_B * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q * BLOCK_SIZE_B))
    offs_pre_out2n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_pre_out2q = (pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q))
    
    out2 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_Q, BLOCK_SIZE_B), dtype=tl.float32)
    for b2 in range(0, B):
        out1_ptrs = out1_ptr + (offs_out1n[:, None] * stride_out1n + offs_r[None, :] * stride_out1r + b2 * stride_out1b2)
        w2_bfly_ptrs = w2_bfly_ptr + (offs_r[:, None] * stride_w2_bflyr + offs_q[None, :] * stride_w2_bflyq + b2 * stride_w2_bflyb2)
        accumulator_in = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_Q), dtype=tl.float32)
        for r in range(0, tl.cdiv(R, BLOCK_SIZE_R)):
            out1_mask = (offs_r[None, :] < (R - r * BLOCK_SIZE_R))
            w2_bfly_mask = (offs_r[:, None] < (R - r * BLOCK_SIZE_R))
            out1 = tl.load(out1_ptrs, mask=out1_mask, other=0.0)
            w2_bfly = tl.load(w2_bfly_ptrs, mask=w2_bfly_mask, other=0.0)
            accumulator_in = tl.dot(out1, w2_bfly, accumulator_in, allow_tf32=False)
            out1_ptrs += BLOCK_SIZE_R * stride_out1r
            w2_bfly_ptrs += BLOCK_SIZE_R * stride_w2_bflyr

        temp = accumulator_in
        pre_out2_ptrs = pre_out2_ptr + (stride_pre_out2n * offs_pre_out2n[:, None] + stride_pre_out2q * offs_pre_out2q[None, :] + b2 * stride_pre_out2b2)
        pre_out2_mask = ((offs_pre_out2n[:, None] < N) & (offs_pre_out2q[None, :] < Q))
        tl.store(pre_out2_ptrs, temp, mask=pre_out2_mask)
        
        b2_mask = tl.arange(0, BLOCK_SIZE_B) == b2

        out2 = tl.where(
            tl.broadcast_to(b2_mask[None, None, :], (BLOCK_SIZE_N, BLOCK_SIZE_Q, BLOCK_SIZE_B)), 
            tl.broadcast_to(temp[:, :, None], (BLOCK_SIZE_N, BLOCK_SIZE_Q, BLOCK_SIZE_B)), 
            out2
        )

    out2 = out2.reshape((BLOCK_SIZE_N, BLOCK_SIZE_Q * BLOCK_SIZE_B))
    out2_ptrs = out2_ptr + (stride_out2n * offs_out2n[:, None] + stride_out2qb2 * offs_out2qb2[None, :])
    out2_mask = ((offs_out2n[:, None] < N) & (offs_out2qb2[None, :] < (Q * B)))
    tl.store(out2_ptrs, out2, mask=out2_mask)

@triton.autotune(configs=_get_triton_monarch_left_autotune_config(), key=['N', 'Q', 'B', 'R'])
@triton.jit
def _triton_monarch_left_kernel_fp16(
    out1_ptr, w2_bfly_ptr, out2_ptr,
    N, R, B, Q,
    stride_out1b2, stride_out1n, stride_out1r,
    stride_w2_bflyb2, stride_w2_bflyr, stride_w2_bflyq,
    stride_out2n, stride_out2qb2,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_Q: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr
    ):

    """
    Computes a batched matrix multiplication between out1 and w2_bfly. The resulting product is stored in out2_ptr 
    (reordered with strided memory layout). Uses FP16 inputs and outputs. Useful for performance benchmarking. Assumes 
    innermost dimension of w2_bfly is contiguous along the per-block rank, and that innermost dimension of out2 is 
    contiguous along the block (not the per-block rank)

    Parameters:
    ----------
    out1_ptr : tl.tensor
        Pointer to the input matrix out1, shape (B, N, R)
    w2_bfly_ptr : tl.tensor
        Pointer to butterfly weight matrix w2_bfly, shape (B, R, Q)
    out2_ptr : tl.tensor
        Pointer to output matrix out2, shape (N, Q * B)

    N, R, B, Q : int
        Matrix dimensions:
        - N: Number of rows in out1
        - R: Product of per-block rank and B
        - B: Number of blocks
        - Q: Number of output features per block

    stride_out1b2, stride_out1n, stride_out1r : int
        Strides for accessing out1_ptr
    stride_w2_bflyb2, stride_w2_bflyr, stride_w2_bflyq : int
        Strides for accessing w2_bfly_ptr
    stride_out2n, stride_out2qb2 : int
        Strides for accessing out2_ptr

    BLOCK_SIZE_N, BLOCK_SIZE_R, BLOCK_SIZE_B, BLOCK_SIZE_Q : tl.constexpr
        Block sizes used for tiling computations
    GROUP_SIZE_N : tl.constexpr
        Number of program IDs grouped together to improve memory locality
    """

    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N) 
    num_pid_q = tl.cdiv(Q, BLOCK_SIZE_Q)
    num_pid_in_group = GROUP_SIZE_N * num_pid_q
    group_id = pid // num_pid_in_group
    first_pid_n = group_id * GROUP_SIZE_N
    group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_N)
    pid_n = first_pid_n + ((pid % num_pid_in_group) % group_size_n)
    pid_q = (pid % num_pid_in_group) // group_size_n

    offs_out1n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_q = (pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) % Q
    offs_r = tl.arange(0, BLOCK_SIZE_R)
    
    offs_out2n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_out2qb2 = (pid_q * BLOCK_SIZE_B * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q * BLOCK_SIZE_B))
    
    out2 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_Q, BLOCK_SIZE_B), dtype=tl.bfloat16)
    for b2 in range(0, B):
        out1_ptrs = out1_ptr + (offs_out1n[:, None] * stride_out1n + offs_r[None, :] * stride_out1r + b2 * stride_out1b2)
        w2_bfly_ptrs = w2_bfly_ptr + (offs_r[:, None] * stride_w2_bflyr + offs_q[None, :] * stride_w2_bflyq + b2 * stride_w2_bflyb2)
        accumulator_in = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_Q), dtype=tl.float32)
        for r in range(0, tl.cdiv(R, BLOCK_SIZE_R)):
            out1_mask = (offs_r[None, :] < (R - r * BLOCK_SIZE_R))
            w2_bfly_mask = (offs_r[:, None] < (R - r * BLOCK_SIZE_R))
            out1 = tl.load(out1_ptrs, mask=out1_mask, other=0.0)
            w2_bfly = tl.load(w2_bfly_ptrs, mask=w2_bfly_mask, other=0.0)
            accumulator_in = tl.dot(out1, w2_bfly, accumulator_in)
            out1_ptrs += BLOCK_SIZE_R * stride_out1r
            w2_bfly_ptrs += BLOCK_SIZE_R * stride_w2_bflyr

        temp = accumulator_in.to(tl.bfloat16)   
        b2_mask = tl.arange(0, BLOCK_SIZE_B) == b2

        out2 = tl.where(
            tl.broadcast_to(b2_mask[None, None, :], (BLOCK_SIZE_N, BLOCK_SIZE_Q, BLOCK_SIZE_B)), 
            tl.broadcast_to(temp[:, :, None], (BLOCK_SIZE_N, BLOCK_SIZE_Q, BLOCK_SIZE_B)), 
            out2
        )

    out2 = out2.reshape((BLOCK_SIZE_N, BLOCK_SIZE_Q * BLOCK_SIZE_B))
    out2_ptrs = out2_ptr + (stride_out2n * offs_out2n[:, None] + stride_out2qb2 * offs_out2qb2[None, :])
    out2_mask = ((offs_out2n[:, None] < N) & (offs_out2qb2[None, :] < (Q * B)))
    tl.store(out2_ptrs, out2, mask=out2_mask)

@triton.jit
def _triton_monarch_left_kernel_fp16_no_autotune(
    out1_ptr, w2_bfly_ptr, out2_ptr,
    N, R, B, Q,
    stride_out1b2, stride_out1n, stride_out1r,
    stride_w2_bflyb2, stride_w2_bflyr, stride_w2_bflyq,
    stride_out2n, stride_out2qb2,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_Q: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr
    ):

    """
    Computes a batched matrix multiplication between out1 and w2_bfly. The resulting product is stored in out2_ptr 
    (reordered with strided memory layout). Uses FP16 inputs and outputs. Useful for performance benchmarking. Assumes 
    innermost dimension of w2_bfly is contiguous along the per-block rank, and that innermost dimension of out2 is 
    contiguous along the block (not the per-block rank)

    Parameters:
    ----------
    out1_ptr : tl.tensor
        Pointer to the input matrix out1, shape (B, N, R)
    w2_bfly_ptr : tl.tensor
        Pointer to butterfly weight matrix w2_bfly, shape (B, R, Q)
    out2_ptr : tl.tensor
        Pointer to output matrix out2, shape (N, Q * B)

    N, R, B, Q : int
        Matrix dimensions:
        - N: Number of rows in out1
        - R: Product of per-block rank and B
        - B: Number of blocks
        - Q: Number of output features per block

    stride_out1b2, stride_out1n, stride_out1r : int
        Strides for accessing out1_ptr
    stride_w2_bflyb2, stride_w2_bflyr, stride_w2_bflyq : int
        Strides for accessing w2_bfly_ptr
    stride_out2n, stride_out2qb2 : int
        Strides for accessing out2_ptr

    BLOCK_SIZE_N, BLOCK_SIZE_R, BLOCK_SIZE_B, BLOCK_SIZE_Q : tl.constexpr
        Block sizes used for tiling computations
    GROUP_SIZE_N : tl.constexpr
        Number of program IDs grouped together to improve memory locality
    """

    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N) 
    num_pid_q = tl.cdiv(Q, BLOCK_SIZE_Q)
    num_pid_in_group = GROUP_SIZE_N * num_pid_q
    group_id = pid // num_pid_in_group
    first_pid_n = group_id * GROUP_SIZE_N
    group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_N)
    pid_n = first_pid_n + ((pid % num_pid_in_group) % group_size_n)
    pid_q = (pid % num_pid_in_group) // group_size_n

    offs_out1n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_q = (pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) % Q
    offs_r = tl.arange(0, BLOCK_SIZE_R)
    
    offs_out2n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_out2qb2 = (pid_q * BLOCK_SIZE_B * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q * BLOCK_SIZE_B))
    
    out2 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_Q, BLOCK_SIZE_B), dtype=tl.bfloat16)
    for b2 in range(0, B):
        out1_ptrs = out1_ptr + (offs_out1n[:, None] * stride_out1n + offs_r[None, :] * stride_out1r + b2 * stride_out1b2)
        w2_bfly_ptrs = w2_bfly_ptr + (offs_r[:, None] * stride_w2_bflyr + offs_q[None, :] * stride_w2_bflyq + b2 * stride_w2_bflyb2)
        accumulator_in = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_Q), dtype=tl.float32)
        for r in range(0, tl.cdiv(R, BLOCK_SIZE_R)):
            out1_mask = (offs_r[None, :] < (R - r * BLOCK_SIZE_R))
            w2_bfly_mask = (offs_r[:, None] < (R - r * BLOCK_SIZE_R))
            out1 = tl.load(out1_ptrs, mask=out1_mask, other=0.0)
            w2_bfly = tl.load(w2_bfly_ptrs, mask=w2_bfly_mask, other=0.0)
            accumulator_in = tl.dot(out1, w2_bfly, accumulator_in)
            out1_ptrs += BLOCK_SIZE_R * stride_out1r
            w2_bfly_ptrs += BLOCK_SIZE_R * stride_w2_bflyr

        temp = accumulator_in.to(tl.bfloat16)   
        b2_mask = tl.arange(0, BLOCK_SIZE_B) == b2

        out2 = tl.where(
            tl.broadcast_to(b2_mask[None, None, :], (BLOCK_SIZE_N, BLOCK_SIZE_Q, BLOCK_SIZE_B)), 
            tl.broadcast_to(temp[:, :, None], (BLOCK_SIZE_N, BLOCK_SIZE_Q, BLOCK_SIZE_B)), 
            out2
        )

    out2 = out2.reshape((BLOCK_SIZE_N, BLOCK_SIZE_Q * BLOCK_SIZE_B))
    out2_ptrs = out2_ptr + (stride_out2n * offs_out2n[:, None] + stride_out2qb2 * offs_out2qb2[None, :])
    out2_mask = ((offs_out2n[:, None] < N) & (offs_out2qb2[None, :] < (Q * B)))
    tl.store(out2_ptrs, out2, mask=out2_mask)

""" Triton Monarch Kernel Launchers """
def _triton_monarch_right_launcher_fp32(
    x: torch.Tensor, 
    w1_bfly: torch.Tensor,
    best_config: triton.Config) -> (torch.Tensor, torch.Tensor):

    """
    Launches the triton_monarch_right_kernel_fp32 Triton kernel

    Parameters:
    ----------
    x : torch.Tensor
        Input matrix of shape (B1, N, P)
    w1_bfly : torch.Tensor
        Weight matrix of shape (B1, P, R), must be contiguous

    Returns:
    -------
    out1 : torch.Tensor
        The accumulated output matrix of shape (B2, N, B1 * (R // B1))
    pre_out1 : torch.Tensor
        The intermediate result matrix of shape (B1, N, B2 * (R // B2))
    """

    assert x.shape[0] == w1_bfly.shape[0], "Incompatible dimensions"
    assert x.shape[2] == w1_bfly.shape[1], "Incompatible dimensions"
    assert w1_bfly.is_contiguous()
    assert x.dtype == torch.float32
    assert w1_bfly.dtype == torch.float32

    B, N, P = x.shape
    B, P, R = w1_bfly.shape
    PER_BLOCK_R = int(R / B)
    out1 = torch.empty((B, N, R), device=x.device, dtype=x.dtype)
    pre_out1 = torch.empty((B, N, R), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(N, best_config.kwargs['BLOCK_SIZE_N']) * triton.cdiv(R, best_config.kwargs['BLOCK_SIZE_R']), B)
    _triton_monarch_right_kernel_fp32[grid](
        x, w1_bfly, out1, pre_out1,
        N, P, B, R, PER_BLOCK_R,
        x.stride(0), x.stride(1), x.stride(2),
        w1_bfly.stride(0), w1_bfly.stride(1), w1_bfly.stride(2),
        out1.stride(0), out1.stride(1), out1.stride(2),
        pre_out1.stride(0), pre_out1.stride(1), pre_out1.stride(2),
        BLOCK_SIZE_N=best_config.kwargs['BLOCK_SIZE_N'],
        BLOCK_SIZE_R=best_config.kwargs['BLOCK_SIZE_R'],
        BLOCK_SIZE_P=best_config.kwargs['BLOCK_SIZE_P'],
        GROUP_SIZE_N=best_config.kwargs['GROUP_SIZE_N'],
        BLOCK_SIZE_B=next_power_of_2(B),
        num_warps=best_config.num_warps,
        num_stages=best_config.num_stages
    )
    return out1, pre_out1

def _triton_monarch_right_launcher_fp16(
    x: torch.Tensor, 
    w1_bfly: torch.Tensor,
    config: triton.Config = None) -> torch.Tensor:

    """
    Launches the triton_monarch_right_kernel_fp16 Triton kernel

    Parameters:
    ----------
    x : torch.Tensor
        Input matrix of shape (B1, N, P)
    w1_bfly : torch.Tensor
        Weight matrix of shape (B1, P, R), must be contiguous

    Returns:
    -------
    out1 : torch.Tensor
        The accumulated output matrix of shape (B2, N, B1 * (R // B1))
    pre_out1 : torch.Tensor
        The intermediate result matrix of shape (B1, N, B2 * (R // B2))
    """

    assert x.shape[0] == w1_bfly.shape[0], "Incompatible dimensions"
    assert x.shape[2] == w1_bfly.shape[1], "Incompatible dimensions"
    assert w1_bfly.is_contiguous()
    assert x.dtype == torch.bfloat16
    assert w1_bfly.dtype == torch.bfloat16

    B, N, P = x.shape
    B, P, R = w1_bfly.shape
    PER_BLOCK_R = int(R / B)
    out1 = torch.empty((B, N, R), device=x.device, dtype=x.dtype)
    if config is None:
        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']) * triton.cdiv(R, META['BLOCK_SIZE_R']), B)
        _triton_monarch_right_kernel_fp16[grid](
            x, w1_bfly, out1,
            N, P, B, R, PER_BLOCK_R,
            x.stride(0), x.stride(1), x.stride(2),
            w1_bfly.stride(0), w1_bfly.stride(1), w1_bfly.stride(2),
            out1.stride(0), out1.stride(1), out1.stride(2),
            BLOCK_SIZE_B=next_power_of_2(B)
        )
    else:
        grid = (triton.cdiv(N, config.kwargs['BLOCK_SIZE_N']) * triton.cdiv(R, config.kwargs['BLOCK_SIZE_R']), B)
        _triton_monarch_right_kernel_fp16_no_autotune[grid](
            x, w1_bfly, out1,
            N, P, B, R, PER_BLOCK_R,
            x.stride(0), x.stride(1), x.stride(2),
            w1_bfly.stride(0), w1_bfly.stride(1), w1_bfly.stride(2),
            out1.stride(0), out1.stride(1), out1.stride(2),
            BLOCK_SIZE_N=config.kwargs['BLOCK_SIZE_N'],
            BLOCK_SIZE_R=config.kwargs['BLOCK_SIZE_R'],
            BLOCK_SIZE_P=config.kwargs['BLOCK_SIZE_P'],
            GROUP_SIZE_N=config.kwargs['GROUP_SIZE_N'],
            BLOCK_SIZE_B=next_power_of_2(B),
            num_warps=config.num_warps,
            num_stages=config.num_stages
        )
    return out1

def _triton_monarch_left_launcher_fp32(
    out1: torch.Tensor, 
    w2_bfly: torch.Tensor,
    best_config: triton.Config) -> (torch.Tensor, torch.Tensor):

    """
    Launches the triton_monarch_left_kernel_fp32 Triton kernel

    Parameters:
    ----------
    out1 : torch.Tensor
        Input matrix of shape (B2, N, B1 * (R // B2))
    w2_bfly : torch.Tensor
        Weight matrix of shape (B2, B1 * (R // B1), Q), must be contiguous

    Returns:
    -------
    out2 : torch.Tensor
        The accumulated output matrix of shape (N, Q * B2)
    pre_out2 : torch.Tensor
        The intermediate result matrix of shape (B2, N, Q)
    """

    assert out1.shape[0] == w2_bfly.shape[0], "Incompatible dimensions"
    assert out1.shape[2] == w2_bfly.shape[1], "Incompatible dimensions"
    assert w2_bfly.is_contiguous()
    assert out1.dtype == torch.float32
    assert w2_bfly.dtype == torch.float32

    B, N, R = out1.shape
    B, R, Q = w2_bfly.shape
    out2 = torch.empty((N, B * Q), device=out1.device, dtype=out1.dtype)
    pre_out2 = torch.empty((B, N, Q), device=out1.device, dtype=out1.dtype)
    grid = (triton.cdiv(N, best_config.kwargs['BLOCK_SIZE_N']) * triton.cdiv(Q, best_config.kwargs['BLOCK_SIZE_Q']), )
    _triton_monarch_left_kernel_fp32[grid](
        out1, w2_bfly, out2, pre_out2,
        N, R, B, Q,
        out1.stride(0), out1.stride(1), out1.stride(2),
        w2_bfly.stride(0), w2_bfly.stride(1), w2_bfly.stride(2),
        out2.stride(0), out2.stride(1),
        pre_out2.stride(0), pre_out2.stride(1), pre_out2.stride(2),
        BLOCK_SIZE_N=best_config.kwargs['BLOCK_SIZE_N'],
        BLOCK_SIZE_R=best_config.kwargs['BLOCK_SIZE_R'],
        BLOCK_SIZE_Q=best_config.kwargs['BLOCK_SIZE_Q'],
        GROUP_SIZE_N=best_config.kwargs['GROUP_SIZE_N'],
        BLOCK_SIZE_B=next_power_of_2(B),
        num_warps=best_config.num_warps,
        num_stages=best_config.num_stages
    )
    return out2, pre_out2

def _triton_monarch_left_launcher_fp16(
    out1: torch.Tensor, 
    w2_bfly: torch.Tensor,
    config: triton.Config = None) -> torch.Tensor:
    
    """
    Launches the triton_monarch_left_kernel_fp16 Triton kernel

    Parameters:
    ----------
    out1 : torch.Tensor
        Input matrix of shape (B2, N, B1 * (R // B2))
    w2_bfly : torch.Tensor
        Weight matrix of shape (B2, B1 * (R // B1), Q), must be contiguous

    Returns:
    -------
    out2 : torch.Tensor
        The accumulated output matrix of shape (N, Q * B2)
    pre_out2 : torch.Tensor
        The intermediate result matrix of shape (B2, N, Q)
    """
    
    assert out1.shape[0] == w2_bfly.shape[0], "Incompatible dimensions"
    assert out1.shape[2] == w2_bfly.shape[1], "Incompatible dimensions"
    assert w2_bfly.is_contiguous()
    assert out1.dtype == torch.bfloat16
    assert w2_bfly.dtype == torch.bfloat16

    B, N, R = out1.shape
    B, R, Q = w2_bfly.shape
    out2 = torch.empty((N, B * Q), device=out1.device, dtype=out1.dtype)
    if config is None:
        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']) * triton.cdiv(Q, META['BLOCK_SIZE_Q']), )
        _triton_monarch_left_kernel_fp16[grid](
            out1, w2_bfly, out2,
            N, R, B, Q,
            out1.stride(0), out1.stride(1), out1.stride(2),
            w2_bfly.stride(0), w2_bfly.stride(1), w2_bfly.stride(2),
            out2.stride(0), out2.stride(1),
            BLOCK_SIZE_B=next_power_of_2(B)
        )
    else:
        grid = (triton.cdiv(N, config.kwargs['BLOCK_SIZE_N']) * triton.cdiv(Q, config.kwargs['BLOCK_SIZE_Q']), )
        _triton_monarch_left_kernel_fp16_no_autotune[grid](
            out1, w2_bfly, out2,
            N, R, B, Q,
            out1.stride(0), out1.stride(1), out1.stride(2),
            w2_bfly.stride(0), w2_bfly.stride(1), w2_bfly.stride(2),
            out2.stride(0), out2.stride(1),
            BLOCK_SIZE_N=config.kwargs['BLOCK_SIZE_N'],
            BLOCK_SIZE_R=config.kwargs['BLOCK_SIZE_R'],
            BLOCK_SIZE_Q=config.kwargs['BLOCK_SIZE_Q'],
            GROUP_SIZE_N=config.kwargs['GROUP_SIZE_N'],
            BLOCK_SIZE_B=next_power_of_2(B),
            num_warps=config.num_warps,
            num_stages=config.num_stages
        )
    return out2

""" Triton Monarch Functions """
def triton_monarch_right_fp32(
    x: torch.Tensor,
    w1_bfly: torch.Tensor,
    w2_bfly_t: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):

    """
    Performs Monarch matrix multiplication using triton_monarch_right_launcher_fp32 and is
    useful for checking correctness of entire Monarch operation. Assumes innermost dimension 
    of w1_bfly is contiguous along the per-block rank.

    Parameters:
    ----------
    x : torch.Tensor
        Input tensor of shape (num_batches, num_seq, in_f)
    w1_bfly : torch.Tensor
        Tensor of shape (b1, in_f // b1, b2 * (r // b2))
    w2_bfly_t : torch.Tensor
        Tensor of shape (b2, out_f // b2, b1 * (r // b1))

    Returns:
    -------
    out: torch.Tensor
        Output tensor of shape (num_batches, num_seq, out_f), resulting from the Monarch matrix multiplication
    """

    best_config = getattr(_triton_monarch_right_kernel_fp16, 'best_config', None)

    if best_config is None:
        default_config = triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8},  num_stages=4, num_warps=4)
        print(f"No autotuned config found. Running with default config: {default_config}")
        print("Ensure to run triton_monarch_right_fp16 before")
        best_config = default_config

    batch_shape, n = x.shape[:-1], x.shape[-1]
    batch_dim = np.prod(batch_shape)
    k, p, q = w1_bfly.shape
    l, s, r = w2_bfly_t.shape
    assert k * p == n
    assert l * r == k * q
    x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
    out1, pre_out1 = _triton_monarch_right_launcher_fp32(x_reshaped, w1_bfly, best_config)
    out2 = torch.empty(batch_dim, l, s, device=x.device, dtype=x.dtype).transpose(0, 1)
    out2 = torch.bmm(out1, w2_bfly_t.transpose(-1, -2), out=out2)
    out2 = out2.permute(1, 2, 0).reshape(*batch_shape, s * l)
    return out2, out1, pre_out1

def triton_monarch_right_fp16(
    x: torch.Tensor,
    w1_bfly: torch.Tensor,
    w2_bfly_t: torch.Tensor,
    config: triton.Config = None) -> torch.Tensor:

    """
    Performs Monarch matrix multiplication using triton_monarch_right_launcher_fp16 and is
    useful for benchmarking performance of entire Monarch operation. Assumes innermost dimension 
    of w1_bfly is contiguous along the per-block rank.

    Parameters:
    ----------
    x : torch.Tensor
        Input tensor of shape (num_batches, num_seq, in_f)
    w1_bfly : torch.Tensor
        Tensor of shape (b1, in_f // b1, b2 * (r // b2))
    w2_bfly_t : torch.Tensor
        Tensor of shape (b2, out_f // b2, b1 * (r // b1))

    Returns:
    -------
    out: torch.Tensor
        Output tensor of shape (num_batches, num_seq, out_f), resulting from the Monarch matrix multiplication
    """

    batch_shape, n = x.shape[:-1], x.shape[-1]
    k, p, q = w1_bfly.shape
    l, s, r = w2_bfly_t.shape
    assert k * p == x.shape[-1]
    assert l * r == k * q

    x_shape = x.shape
    x = x.flatten(0, -2)
    x = x.view(-1, k, x_shape[-1] // k).transpose(0, 1)

    out1 = _triton_monarch_right_launcher_fp16(x, w1_bfly, config)
    out2 = torch.empty(l, x_shape[0] * x_shape[1], s, device=x.device, dtype=x.dtype)
    out2 = torch.bmm(out1, w2_bfly_t.transpose(-1, -2), out=out2)
    out2 = out2.permute(1, 2, 0).reshape(*batch_shape, s * l)
    return out2

def triton_monarch_right_ideal_fp32(
    x: torch.Tensor,
    w1_bfly: torch.Tensor,
    w2_bfly_t: torch.Tensor) -> torch.Tensor:

    """
    Performs Monarch matrix multiplication using triton_monarch_right_launcher_fp32 and is
    useful for checking correctness of entire Monarch operation. Assumes innermost dimension 
    of w1_bfly is contiguous along the per-block rank. 
    
    Keeps final output innermost dimension fixed unlike triton_monarch_right_fp32. Assumes subsequent 
    layer's weight matrix has innermost dimension contiguous in B1 dimension (only works for non-final 
    MLP layers).

    Parameters:
    ----------
    x : torch.Tensor
        Input tensor of shape (num_batches, num_seq, in_f)
    w1_bfly : torch.Tensor
        Tensor of shape (b1, in_f // b1, b2 * (r // b2))
    w2_bfly_t : torch.Tensor
        Tensor of shape (b2, out_f // b2, b1 * (r // b1))

    Returns:
    -------
    out: torch.Tensor
        Output tensor of shape (num_batches, num_seq, out_f), resulting from the Monarch matrix multiplication
    """

    best_config = getattr(_triton_monarch_right_kernel_fp16, 'best_config', None)

    if best_config is None:
        default_config = triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8},  num_stages=4, num_warps=4)
        print(f"No autotuned config found. Running with default config: {default_config}")
        print("Ensure to run triton_monarch_right_ideal_fp16 before")
        best_config = default_config

    batch_shape, n = x.shape[:-1], x.shape[-1]
    batch_dim = np.prod(batch_shape)
    k, p, q = w1_bfly.shape
    l, s, r = w2_bfly_t.shape
    assert k * p == n
    assert l * r == k * q
    x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
    out1, pre_out1 = _triton_monarch_right_launcher_fp32(x_reshaped, w1_bfly, best_config)
    out2 = torch.empty(batch_dim, l, s, device=x.device, dtype=x.dtype).transpose(0, 1)
    out2 = torch.bmm(out1, w2_bfly_t.transpose(-1, -2), out=out2)
    out2 = out2.permute(1, 0, 2).reshape(*batch_shape, s * l)
    return out2, out1, pre_out1

def triton_monarch_right_ideal_fp16(
    x: torch.Tensor,
    w1_bfly: torch.Tensor,
    w2_bfly_t: torch.Tensor,
    config: triton.Config = None) -> torch.Tensor:

    """
    Performs Monarch matrix multiplication using triton_monarch_right_launcher_fp16 and is
    useful for benchmarking performance of entire Monarch operation. Assumes innermost dimension 
    of w1_bfly is contiguous along the per-block rank. 
    
    Keeps final output innermost dimension fixed unlike triton_monarch_right_fp16. Assumes subsequent 
    layer's weight matrix has innermost dimension contiguous in B1 dimension (only works for non-final 
    MLP layers).

    Parameters:
    ----------
    x : torch.Tensor
        Input tensor of shape (num_batches, num_seq, in_f)
    w1_bfly : torch.Tensor
        Tensor of shape (b1, in_f // b1, b2 * (r // b2))
    w2_bfly_t : torch.Tensor
        Tensor of shape (b2, out_f // b2, b1 * (r // b1))

    Returns:
    -------
    out: torch.Tensor
        Output tensor of shape (num_batches, num_seq, out_f), resulting from the Monarch matrix multiplication
    """

    batch_shape, n = x.shape[:-1], x.shape[-1]
    k, p, q = w1_bfly.shape
    l, s, r = w2_bfly_t.shape
    assert k * p == n
    assert l * r == k * q

    x_shape = x.shape
    x = x.flatten(0, -2)
    x = x.view(-1, k, x_shape[-1] // k).transpose(0, 1)

    out1 = _triton_monarch_right_launcher_fp16(x, w1_bfly, config)
    out2 = torch.empty(l, x_shape[0] * x_shape[1], s, device=x.device, dtype=x.dtype)
    out2 = torch.bmm(out1, w2_bfly_t.transpose(-1, -2), out=out2)
    out2 = out2.permute(1, 0, 2).reshape(*batch_shape, s * l)
    return out2

def triton_monarch_right_left_fp32(
    x: torch.Tensor,
    w1_bfly: torch.Tensor,
    w2_bfly: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):

    """
    Performs Monarch matrix multiplication using triton_monarch_right_launcher_fp32 and 
    triton_monarch_left_launcher_fp32, useful for checking correctness of entire Monarch operation. 
    Assumes innermost dimension of w1_bfly is contiguous along the per-block rank.

    Parameters:
    ----------
    x : torch.Tensor
        Input tensor of shape (num_batches, num_seq, in_f)
    w1_bfly : torch.Tensor
        Tensor of shape (b1, in_f // b1, b2 * (r // b2))
    w2_bfly : torch.Tensor
        Tensor of shape (b2, b1 * (r // b1), out_f // b2)

    Returns:
    -------
    out: torch.Tensor
        Output tensor of shape (num_batches, num_seq, out_f), resulting from the Monarch matrix multiplication
    """

    best_config_right = getattr(_triton_monarch_right_kernel_fp16, 'best_config', None)

    if best_config_right is None:
        default_config_right = triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8},  num_stages=4, num_warps=4)
        print(f"No autotuned config found. Running with default config: {default_config_right}")
        print("Ensure to run triton_monarch_right_left_fp16 before")
        best_config_right = default_config_right

    best_config_left = getattr(_triton_monarch_left_kernel_fp16, 'best_config', None)

    if best_config_left is None:
        default_config_left = triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_Q': 32,  'BLOCK_SIZE_R': 32,  'GROUP_SIZE_N': 8},  num_stages=3, num_warps=4)
        print(f"No autotuned config found. Running with default config: {default_config_left}")
        print("Ensure to run triton_monarch_right_left_fp16 before")
        best_config_left = default_config_left
    
    batch_shape, n = x.shape[:-1], x.shape[-1]
    batch_dim = np.prod(batch_shape)
    k, p, q = w1_bfly.shape
    l, r, s = w2_bfly.shape
    assert k * p == n
    assert l * r == k * q
    x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
    out1, pre_out1 = _triton_monarch_right_launcher_fp32(x_reshaped, w1_bfly, best_config_right)
    out2, pre_out2 = _triton_monarch_left_launcher_fp32(out1, w2_bfly, best_config_left)
    out2 = out2.reshape(*batch_shape, s * l)
    return out2, pre_out2, out1, pre_out1

def triton_monarch_right_left_fp16(
    x: torch.Tensor,
    w1_bfly: torch.Tensor,
    w2_bfly: torch.Tensor,
    config: (triton.Config, triton.Config) = (None, None)) -> torch.Tensor:

    """
    Performs Monarch matrix multiplication using triton_monarch_right_launcher_fp16 and 
    triton_monarch_left_launcher_fp16, useful for benchmarking performance of entire Monarch operation. 
    Assumes innermost dimension of w1_bfly is contiguous along the per-block rank.

    Parameters:
    ----------
    x : torch.Tensor
        Input tensor of shape (num_batches, num_seq, in_f)
    w1_bfly : torch.Tensor
        Tensor of shape (b1, in_f // b1, b2 * (r // b2))
    w2_bfly : torch.Tensor
        Tensor of shape (b2, b1 * (r // b1), out_f // b2)

    Returns:
    -------
    out: torch.Tensor
        Output tensor of shape (num_batches, num_seq, out_f), resulting from the Monarch matrix multiplication
    """

    batch_shape, n = x.shape[:-1], x.shape[-1]
    k, p, q = w1_bfly.shape
    l, r, s = w2_bfly.shape
    assert k * p == n
    assert l * r == k * q

    x_shape = x.shape
    x = x.flatten(0, -2)
    x = x.view(-1, k, x_shape[-1] // k).transpose(0, 1)

    out1 = _triton_monarch_right_launcher_fp16(x, w1_bfly, config[0])
    out2 = _triton_monarch_left_launcher_fp16(out1, w2_bfly, config[1])
    out2 = out2.reshape(*batch_shape, s * l)
    return out2

""" Get Triton Monarch Kernel Autotuned Configuration """
def get_triton_monarch_right_fp16_config():
    return getattr(_triton_monarch_right_kernel_fp16, 'best_config', None)

def get_triton_monarch_right_ideal_fp16_config():
    return get_triton_monarch_right_fp16_config()

def get_triton_monarch_right_left_fp16_config():
    return (getattr(_triton_monarch_right_kernel_fp16, 'best_config', None),
            getattr(_triton_monarch_left_kernel_fp16, 'best_config', None), )