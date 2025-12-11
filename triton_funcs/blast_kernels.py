import triton
import triton.language as tl
import torch
from utils import next_power_of_2
from typing import Tuple

""" Autotune Configurations for Triton BLAST Kernels """
def _get_triton_blast_partial_kernel_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_P': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_P': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_P': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_P': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_P': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_P': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_P': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_P': 32}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_P': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_P': 32}, num_stages=3, num_warps=4)
    ]

def _get_triton_blast_partial_grouped_kernel_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_R': 16,  'BLOCK_SIZE_P': 64,  'GROUP_SIZE_N': 8}, num_stages=3,  num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_R': 16,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=4,  num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 128, 'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=4,  num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=4,  num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=4,  num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=3,  num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=5,  num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 64,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=3,  num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 64,  'GROUP_SIZE_N': 8}, num_stages=5,  num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=5,  num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 16,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=5,  num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 128, 'GROUP_SIZE_N': 8}, num_stages=3,  num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 128, 'GROUP_SIZE_N': 4}, num_stages=3,  num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 64,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=5,  num_warps=2)
    ]

def _get_triton_blast_partial_grouped_persistent_kernel_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_R': 16,  'BLOCK_SIZE_P': 64,  'GROUP_SIZE_N': 8}, num_stages=3,  num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_R': 16,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=4,  num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 128, 'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=4,  num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=4,  num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=4,  num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=3,  num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=5,  num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 64,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=3,  num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 64,  'GROUP_SIZE_N': 8}, num_stages=5,  num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=5,  num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 16,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=5,  num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 128, 'GROUP_SIZE_N': 8}, num_stages=3,  num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 128, 'GROUP_SIZE_N': 4}, num_stages=3,  num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 64,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=5,  num_warps=2)
    ]

def _get_triton_blast_full_kernel_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_Q': 256, 'BLOCK_SIZE_P': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_Q': 256, 'BLOCK_SIZE_P': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_Q': 128, 'BLOCK_SIZE_P': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_Q': 64,  'BLOCK_SIZE_P': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_Q': 32,  'BLOCK_SIZE_P': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_Q': 256, 'BLOCK_SIZE_P': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_Q': 128, 'BLOCK_SIZE_P': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_Q': 32,  'BLOCK_SIZE_P': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_Q': 32,  'BLOCK_SIZE_P': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_Q': 64,  'BLOCK_SIZE_P': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_Q': 64,  'BLOCK_SIZE_P': 32}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_Q': 64,  'BLOCK_SIZE_P': 32}, num_stages=5, num_warps=2)
    ]

def _get_triton_blast_bmm_xv_kernel_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_R': 128, 'BLOCK_SIZE_P': 64,  'GROUP_SIZE_N': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_R': 256, 'BLOCK_SIZE_P': 64,  'GROUP_SIZE_N': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_R': 128, 'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_R': 64,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 256, 'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 128, 'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 64,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 64,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 64,  'GROUP_SIZE_N': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 128, 'GROUP_SIZE_N': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 128, 'GROUP_SIZE_N': 4}, num_stages=3, num_warps=4),
    ]

def _get_triton_blast_bmm_sxv_kernel_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_N': BLOCK_SIZE_N}, num_stages=num_stages, num_warps=num_warps)
        for BLOCK_SIZE_N in [32, 64, 128, 256, 512]
        for num_stages in [1, 2, 3, 4, 5, 6]
        for num_warps in [2, 4, 8]
    ]

def _get_triton_blast_bmm_usxv_kernel_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_Q': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_R': 64,  'GROUP_SIZE_Q': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_Q': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_R': 64,  'GROUP_SIZE_Q': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_Q': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_R': 32,  'GROUP_SIZE_Q': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_Q': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 32,  'GROUP_SIZE_Q': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_Q': 128, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 32,  'GROUP_SIZE_Q': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_Q': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_R': 32,  'GROUP_SIZE_Q': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_Q': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_R': 32,  'GROUP_SIZE_Q': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_Q': 64,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 32,  'GROUP_SIZE_Q': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_Q': 64,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 32,  'GROUP_SIZE_Q': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_Q': 64,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 32,  'GROUP_SIZE_Q': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_Q': 64,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 32,  'GROUP_SIZE_Q': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_Q': 32,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 32,  'GROUP_SIZE_Q': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_Q': 32,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 32,  'GROUP_SIZE_Q': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_Q': 32,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 64,  'GROUP_SIZE_Q': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_Q': 32,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 32,  'GROUP_SIZE_Q': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_Q': 32,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 128, 'GROUP_SIZE_Q': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_Q': 32,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 128, 'GROUP_SIZE_Q': 4}, num_stages=3, num_warps=4),
    ]

""" Triton BLAST Kernels """
#-----------------------------------
@triton.jit
def _triton_blast_partial_kernel_fp32(
    x_ptr, v_ptr, s_ptr, z_ptr, y_ptr,
    N, P, B1, R, B2,
    stride_xb1, stride_xn, stride_xp,
    stride_vb1, stride_vp, stride_vr,
    stride_sb1, stride_sb2, stride_sr,
    stride_zb2, stride_zn, stride_zr,
    stride_yb1, stride_yn, stride_yr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_P: tl.constexpr, BLOCK_SIZE_B1: tl.constexpr, BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_B2: tl.constexpr):

    """ 
    Computes batched matrix multiplication between X, V, and S, and produces outputs stored 
    in Y and Z, using FP32 inputs and outputs. Useful for checking correctness.

    Parameters:
    ----------
    x_ptr : tl.tensor
        Pointer to the input matrix X, shape (B1, N, P)
    v_ptr : tl.tensor
        Pointer to the input matrix V, shape (B1, P, R)
    s_ptr : tl.tensor
        Pointer to the scaling matrix S, shape (B1, B2, R)
    z_ptr : tl.tensor
        Pointer to the output matrix Z, shape (B2, N, R)
    y_ptr : tl.tensor
        Pointer to the intermediate output matrix Y, shape (B1, N, R)

    N, P, B1, R, B2 : int
        Dimensions of the input and output tensors

    stride_xb1, stride_xn, stride_xp : int
        Strides for indexing into x_ptr
    stride_vb1, stride_vp, stride_vr : int
        Strides for indexing into v_ptr
    stride_sb1, stride_sb2, stride_sr : int
        Strides for indexing into s_ptr
    stride_zb2, stride_zn, stride_zr : int
        Strides for indexing into z_ptr
    stride_yb1, stride_yn, stride_yr : int
        Strides for indexing into y_ptr

    BLOCK_SIZE_N, BLOCK_SIZE_P, BLOCK_SIZE_B1, BLOCK_SIZE_R, BLOCK_SIZE_B2 : tl.constexpr
        Constants defining the block sizes for partitioning the computation
    """

    pid_n = tl.program_id(axis=0)
    offs_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_p = tl.arange(0, BLOCK_SIZE_P)
    offs_r = tl.arange(0, BLOCK_SIZE_R)
    offs_b2 = tl.arange(0, BLOCK_SIZE_B2)

    accumulator_out = tl.zeros((BLOCK_SIZE_B2, BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
    for b1 in range(0, B1):
        x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_p[None, :] * stride_xp + b1 * stride_xb1)
        v_ptrs = v_ptr + (offs_p[:, None] * stride_vp + offs_r[None, :] * stride_vr + b1 * stride_vb1)
        s_ptrs = s_ptr + (offs_b2[:, None] * stride_sb2 + offs_r[None, :] * stride_sr + b1 * stride_sb1)
        y_ptrs = y_ptr + (stride_yn * offs_yn[:, None] + stride_yr * offs_r[None, :] + b1 * stride_yb1)
        accumulator_in = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
        s_mask = ((offs_b2[:, None] < B2) & (offs_r[None, :] < R))
        s = tl.load(s_ptrs, mask = s_mask, other=0.0)
        s = tl.expand_dims(s, 1)
        for p in range(0, tl.cdiv(P, BLOCK_SIZE_P)):
            x_mask = ((offs_p[None, :] < (P - p * BLOCK_SIZE_P)) & (offs_xn[:, None] < N))
            v_mask = ((offs_p[:, None] < (P - p * BLOCK_SIZE_P)) & (offs_r[None, :] < R))
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            accumulator_in = tl.dot(x, v, accumulator_in, allow_tf32=False)
            x_ptrs += BLOCK_SIZE_P * stride_xp
            v_ptrs += BLOCK_SIZE_P * stride_vp
        y = accumulator_in
        y_mask = ((offs_yn[:, None] < N) & (offs_r[None, :] < R))
        tl.store(y_ptrs, y, mask=y_mask)

        y = tl.expand_dims(y, 0)
        accumulator_out += s * y

    z = accumulator_out
    offs_zn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    z_ptrs = z_ptr + (stride_zn * offs_zn[None, :, None] + stride_zr * offs_r[None, None, :] + stride_zb2 * offs_b2[:, None, None])
    z_mask = ((offs_zn[None, :, None] < N) & (offs_r[None, None, :] < R) & (offs_b2[:, None, None] < B2))
    tl.store(z_ptrs, z, mask=z_mask)

@triton.autotune(configs=_get_triton_blast_partial_kernel_autotune_config(), key=['N', 'P', 'R', 'B1', 'B2'])
@triton.jit
def _triton_blast_partial_kernel_fp16(
    x_ptr, v_ptr, s_ptr, z_ptr,
    N, P, B1, R, B2,
    stride_xb1, stride_xn, stride_xp,
    stride_vb1, stride_vp, stride_vr,
    stride_sb1, stride_sb2, stride_sr,
    stride_zb2, stride_zn, stride_zr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_P: tl.constexpr, BLOCK_SIZE_B1: tl.constexpr, BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_B2: tl.constexpr
    ):

    """ 
    Computes batched matrix multiplication between X, V, and S, and produces outputs stored 
    in Y and Z, using FP16 inputs and outputs. Useful for performance benchmarking.

    Parameters:
    ----------
    x_ptr : tl.tensor
        Pointer to the input matrix X, shape (B1, N, P)
    v_ptr : tl.tensor
        Pointer to the input matrix V, shape (B1, P, R)
    s_ptr : tl.tensor
        Pointer to the scaling matrix S, shape (B1, B2, R)
    z_ptr : tl.tensor
        Pointer to the output matrix Z, shape (B2, N, R)

    N, P, B1, R, B2 : int
        Dimensions of the input and output tensors

    stride_xb1, stride_xn, stride_xp : int
        Strides for indexing into x_ptr
    stride_vb1, stride_vp, stride_vr : int
        Strides for indexing into v_ptr
    stride_sb1, stride_sb2, stride_sr : int
        Strides for indexing into s_ptr
    stride_zb2, stride_zn, stride_zr : int
        Strides for indexing into z_ptr

    BLOCK_SIZE_N, BLOCK_SIZE_P, BLOCK_SIZE_B1, BLOCK_SIZE_R, BLOCK_SIZE_B2 : tl.constexpr
        Constants defining the block sizes for partitioning the computation
    """

    pid_n = tl.program_id(axis=0)
    offs_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_p = tl.arange(0, BLOCK_SIZE_P)
    offs_r = tl.arange(0, BLOCK_SIZE_R)
    offs_b2 = tl.arange(0, BLOCK_SIZE_B2)

    accumulator_out = tl.zeros((BLOCK_SIZE_B2, BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
    for b1 in range(0, B1):
        x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_p[None, :] * stride_xp + b1 * stride_xb1)
        v_ptrs = v_ptr + (offs_p[:, None] * stride_vp + offs_r[None, :] * stride_vr + b1 * stride_vb1)
        s_ptrs = s_ptr + (offs_b2[:, None] * stride_sb2 + offs_r[None, :] * stride_sr + b1 * stride_sb1)
        accumulator_in = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
        s_mask = ((offs_b2[:, None] < B2) & (offs_r[None, :] < R))
        s = tl.load(s_ptrs, mask = s_mask, other=0.0)
        s = tl.expand_dims(s, 1)
        for p in range(0, tl.cdiv(P, BLOCK_SIZE_P)):
            x_mask = ((offs_p[None, :] < (P - p * BLOCK_SIZE_P)) & (offs_xn[:, None] < N))
            v_mask = ((offs_p[:, None] < (P - p * BLOCK_SIZE_P)) & (offs_r[None, :] < R))
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            accumulator_in = tl.dot(x, v, accumulator_in)
            x_ptrs += BLOCK_SIZE_P * stride_xp
            v_ptrs += BLOCK_SIZE_P * stride_vp
        y = accumulator_in.to(tl.bfloat16)

        y = tl.expand_dims(y, 0)
        accumulator_out += s * y

    z = accumulator_out.to(tl.bfloat16)
    offs_zn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    z_ptrs = z_ptr + (stride_zn * offs_zn[None, :, None] + stride_zr * offs_r[None, None, :] + stride_zb2 * offs_b2[:, None, None])
    z_mask = ((offs_zn[None, :, None] < N) & (offs_r[None, None, :] < R) & (offs_b2[:, None, None] < B2))
    tl.store(z_ptrs, z, mask=z_mask)

@triton.jit
def _triton_blast_partial_kernel_fp16_no_autotune(
    x_ptr, v_ptr, s_ptr, z_ptr,
    N, P, B1, R, B2,
    stride_xb1, stride_xn, stride_xp,
    stride_vb1, stride_vp, stride_vr,
    stride_sb1, stride_sb2, stride_sr,
    stride_zb2, stride_zn, stride_zr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_P: tl.constexpr, BLOCK_SIZE_B1: tl.constexpr, BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_B2: tl.constexpr
    ):

    """ 
    Computes batched matrix multiplication between X, V, and S, and produces outputs stored 
    in Y and Z, using FP16 inputs and outputs. Useful for performance benchmarking.

    Parameters:
    ----------
    x_ptr : tl.tensor
        Pointer to the input matrix X, shape (B1, N, P)
    v_ptr : tl.tensor
        Pointer to the input matrix V, shape (B1, P, R)
    s_ptr : tl.tensor
        Pointer to the scaling matrix S, shape (B1, B2, R)
    z_ptr : tl.tensor
        Pointer to the output matrix Z, shape (B2, N, R)

    N, P, B1, R, B2 : int
        Dimensions of the input and output tensors

    stride_xb1, stride_xn, stride_xp : int
        Strides for indexing into x_ptr
    stride_vb1, stride_vp, stride_vr : int
        Strides for indexing into v_ptr
    stride_sb1, stride_sb2, stride_sr : int
        Strides for indexing into s_ptr
    stride_zb2, stride_zn, stride_zr : int
        Strides for indexing into z_ptr

    BLOCK_SIZE_N, BLOCK_SIZE_P, BLOCK_SIZE_B1, BLOCK_SIZE_R, BLOCK_SIZE_B2 : tl.constexpr
        Constants defining the block sizes for partitioning the computation
    """

    pid_n = tl.program_id(axis=0)
    offs_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_p = tl.arange(0, BLOCK_SIZE_P)
    offs_r = tl.arange(0, BLOCK_SIZE_R)
    offs_b2 = tl.arange(0, BLOCK_SIZE_B2)

    accumulator_out = tl.zeros((BLOCK_SIZE_B2, BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
    for b1 in range(0, B1):
        x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_p[None, :] * stride_xp + b1 * stride_xb1)
        v_ptrs = v_ptr + (offs_p[:, None] * stride_vp + offs_r[None, :] * stride_vr + b1 * stride_vb1)
        s_ptrs = s_ptr + (offs_b2[:, None] * stride_sb2 + offs_r[None, :] * stride_sr + b1 * stride_sb1)
        accumulator_in = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
        s_mask = ((offs_b2[:, None] < B2) & (offs_r[None, :] < R))
        s = tl.load(s_ptrs, mask = s_mask, other=0.0)
        s = tl.expand_dims(s, 1)
        for p in range(0, tl.cdiv(P, BLOCK_SIZE_P)):
            x_mask = ((offs_p[None, :] < (P - p * BLOCK_SIZE_P)) & (offs_xn[:, None] < N))
            v_mask = ((offs_p[:, None] < (P - p * BLOCK_SIZE_P)) & (offs_r[None, :] < R))
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            accumulator_in = tl.dot(x, v, accumulator_in)
            x_ptrs += BLOCK_SIZE_P * stride_xp
            v_ptrs += BLOCK_SIZE_P * stride_vp
        y = accumulator_in.to(tl.bfloat16)

        y = tl.expand_dims(y, 0)
        accumulator_out += s * y

    z = accumulator_out.to(tl.bfloat16)
    offs_zn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    z_ptrs = z_ptr + (stride_zn * offs_zn[None, :, None] + stride_zr * offs_r[None, None, :] + stride_zb2 * offs_b2[:, None, None])
    z_mask = ((offs_zn[None, :, None] < N) & (offs_r[None, None, :] < R) & (offs_b2[:, None, None] < B2))
    tl.store(z_ptrs, z, mask=z_mask)

#-----------------------------------
@triton.jit
def _triton_blast_partial_grouped_kernel_fp32(
    x_ptr, v_ptr, s_ptr, z_ptr, y_ptr,
    N, P, B1, R, B2,
    stride_xb1, stride_xn, stride_xp,
    stride_vb1, stride_vp, stride_vr,
    stride_sb1, stride_sb2, stride_sr,
    stride_zb2, stride_zn, stride_zr,
    stride_yb1, stride_yn, stride_yr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_P: tl.constexpr, BLOCK_SIZE_B1: tl.constexpr, BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_B2: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr):

    """ 
    Computes batched matrix multiplication between X, V, and S, and produces outputs stored 
    in Y and Z, using FP32 inputs and outputs. Useful for checking correctness. Promotes better 
    data reuse by super-grouping blocks in groups of GROUP_N rows before switching to the next column.

    Parameters:
    ----------
    x_ptr : tl.tensor
        Pointer to the input matrix X, shape (B1, N, P)
    v_ptr : tl.tensor
        Pointer to the input matrix V, shape (B1, P, R)
    s_ptr : tl.tensor
        Pointer to the scaling matrix S, shape (B1, B2, R)
    z_ptr : tl.tensor
        Pointer to the output matrix Z, shape (B2, N, R)
    y_ptr : tl.tensor
        Pointer to the intermediate output matrix Y, shape (B1, N, R)

    N, P, B1, R, B2 : int
        Dimensions of the input and output tensors

    stride_xb1, stride_xn, stride_xp : int
        Strides for indexing into x_ptr
    stride_vb1, stride_vp, stride_vr : int
        Strides for indexing into v_ptr
    stride_sb1, stride_sb2, stride_sr : int
        Strides for indexing into s_ptr
    stride_zb2, stride_zn, stride_zr : int
        Strides for indexing into z_ptr
    stride_yb1, stride_yn, stride_yr : int
        Strides for indexing into y_ptr

    BLOCK_SIZE_N, BLOCK_SIZE_P, BLOCK_SIZE_B1, BLOCK_SIZE_R, BLOCK_SIZE_B2 : tl.constexpr
        Constants defining the block sizes for partitioning the computation
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

    offs_xn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_r = (pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)) % R
    offs_yn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_yr = (pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R))
    offs_p = tl.arange(0, BLOCK_SIZE_P)
    offs_b2 = tl.arange(0, BLOCK_SIZE_B2)

    accumulator_out = tl.zeros((BLOCK_SIZE_B2, BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)

    for b1 in range(0, B1):
        x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_p[None, :] * stride_xp + b1 * stride_xb1)
        v_ptrs = v_ptr + (offs_p[:, None] * stride_vp + offs_r[None, :] * stride_vr + b1 * stride_vb1)
        s_ptrs = s_ptr + (offs_b2[:, None] * stride_sb2 + offs_r[None, :] * stride_sr + b1 * stride_sb1)
        accumulator_in = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
        s_mask = ((offs_b2[:, None] < B2) & (offs_r[None, :] < R))
        s = tl.load(s_ptrs, mask = s_mask, other=0.0)
        s = tl.expand_dims(s, 1)
        for p in range(0, tl.cdiv(P, BLOCK_SIZE_P)):
            x_mask = (offs_p[None, :] < (P - p * BLOCK_SIZE_P))
            v_mask = (offs_p[:, None] < (P - p * BLOCK_SIZE_P))
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            accumulator_in = tl.dot(x, v, accumulator_in, allow_tf32=False)
            x_ptrs += BLOCK_SIZE_P * stride_xp
            v_ptrs += BLOCK_SIZE_P * stride_vp
        y_ptrs = y_ptr + (stride_yn * offs_yn[:, None] + stride_yr * offs_yr[None, :] + b1 * stride_yb1)
        y = accumulator_in
        y_mask = ((offs_yn[:, None] < N) & (offs_r[None, :] < R))
        tl.store(y_ptrs, y, mask=y_mask)

        y = tl.expand_dims(y, 0)
        accumulator_out += s * y

    z = accumulator_out
    offs_zn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_zr = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    z_ptrs = z_ptr + stride_zn * offs_zn[None, :, None] + stride_zr * offs_zr[None, None, :] + stride_zb2 * offs_b2[:, None, None]
    z_mask = ((offs_zn[None, :, None] < N) & (offs_zr[None, None, :] < R) & (offs_b2[:, None, None] < B2))
    tl.store(z_ptrs, z, mask=z_mask)

@triton.autotune(configs=_get_triton_blast_partial_grouped_kernel_autotune_config(), key=['N', 'P', 'R', 'B1', 'B2'])
@triton.jit
def _triton_blast_partial_grouped_kernel_fp16(
    x_ptr, v_ptr, s_ptr, z_ptr,
    N, P, B1, R, B2,
    stride_xb1, stride_xn, stride_xp,
    stride_vb1, stride_vp, stride_vr,
    stride_sb1, stride_sb2, stride_sr,
    stride_zb2, stride_zn, stride_zr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_P: tl.constexpr, BLOCK_SIZE_B1: tl.constexpr, BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_B2: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr):

    """ 
    Computes batched matrix multiplication between X, V, and S, and produces outputs stored 
    in Y and Z, using FP16 inputs and outputs. Useful for performance benchmarking. Promotes better 
    data reuse by super-grouping blocks in groups of GROUP_N rows before switching to the next column.

    Parameters:
    ----------
    x_ptr : tl.tensor
        Pointer to the input matrix X, shape (B1, N, P)
    v_ptr : tl.tensor
        Pointer to the input matrix V, shape (B1, P, R)
    s_ptr : tl.tensor
        Pointer to the scaling matrix S, shape (B1, B2, R)
    z_ptr : tl.tensor
        Pointer to the output matrix Z, shape (B2, N, R)
    y_ptr : tl.tensor
        Pointer to the intermediate output matrix Y, shape (B1, N, R)

    N, P, B1, R, B2 : int
        Dimensions of the input and output tensors

    stride_xb1, stride_xn, stride_xp : int
        Strides for indexing into x_ptr
    stride_vb1, stride_vp, stride_vr : int
        Strides for indexing into v_ptr
    stride_sb1, stride_sb2, stride_sr : int
        Strides for indexing into s_ptr
    stride_zb2, stride_zn, stride_zr : int
        Strides for indexing into z_ptr
    stride_yb1, stride_yn, stride_yr : int
        Strides for indexing into y_ptr

    BLOCK_SIZE_N, BLOCK_SIZE_P, BLOCK_SIZE_B1, BLOCK_SIZE_R, BLOCK_SIZE_B2 : tl.constexpr
        Constants defining the block sizes for partitioning the computation
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

    offs_xn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_r = (pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)) % R
    offs_p = tl.arange(0, BLOCK_SIZE_P)
    offs_b2 = tl.arange(0, BLOCK_SIZE_B2)

    accumulator_out = tl.zeros((BLOCK_SIZE_B2, BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)

    for b1 in range(0, B1):
        x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_p[None, :] * stride_xp + b1 * stride_xb1)
        v_ptrs = v_ptr + (offs_p[:, None] * stride_vp + offs_r[None, :] * stride_vr + b1 * stride_vb1)
        s_ptrs = s_ptr + (offs_b2[:, None] * stride_sb2 + offs_r[None, :] * stride_sr + b1 * stride_sb1)
        accumulator_in = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
        s_mask = ((offs_b2[:, None] < B2) & (offs_r[None, :] < R))
        s = tl.load(s_ptrs, mask = s_mask, other=0.0)
        s = tl.expand_dims(s, 1)
        for p in range(0, tl.cdiv(P, BLOCK_SIZE_P)):
            x_mask = (offs_p[None, :] < (P - p * BLOCK_SIZE_P))
            v_mask = (offs_p[:, None] < (P - p * BLOCK_SIZE_P))
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            accumulator_in = tl.dot(x, v, accumulator_in)
            x_ptrs += BLOCK_SIZE_P * stride_xp
            v_ptrs += BLOCK_SIZE_P * stride_vp
        y = accumulator_in.to(tl.bfloat16)

        y = tl.expand_dims(y, 0)
        accumulator_out += s * y

    z = accumulator_out.to(tl.bfloat16)
    offs_zn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_zr = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    z_ptrs = z_ptr + stride_zn * offs_zn[None, :, None] + stride_zr * offs_zr[None, None, :] + stride_zb2 * offs_b2[:, None, None]
    z_mask = ((offs_zn[None, :, None] < N) & (offs_zr[None, None, :] < R) & (offs_b2[:, None, None] < B2))
    tl.store(z_ptrs, z, mask=z_mask)

@triton.jit
def _triton_blast_partial_grouped_kernel_fp16_no_autotune(
    x_ptr, v_ptr, s_ptr, z_ptr,
    N, P, B1, R, B2,
    stride_xb1, stride_xn, stride_xp,
    stride_vb1, stride_vp, stride_vr,
    stride_sb1, stride_sb2, stride_sr,
    stride_zb2, stride_zn, stride_zr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_P: tl.constexpr, BLOCK_SIZE_B1: tl.constexpr, BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_B2: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr):

    """ 
    Computes batched matrix multiplication between X, V, and S, and produces outputs stored 
    in Y and Z, using FP16 inputs and outputs. Useful for performance benchmarking. Promotes better 
    data reuse by super-grouping blocks in groups of GROUP_N rows before switching to the next column.

    Parameters:
    ----------
    x_ptr : tl.tensor
        Pointer to the input matrix X, shape (B1, N, P)
    v_ptr : tl.tensor
        Pointer to the input matrix V, shape (B1, P, R)
    s_ptr : tl.tensor
        Pointer to the scaling matrix S, shape (B1, B2, R)
    z_ptr : tl.tensor
        Pointer to the output matrix Z, shape (B2, N, R)
    y_ptr : tl.tensor
        Pointer to the intermediate output matrix Y, shape (B1, N, R)

    N, P, B1, R, B2 : int
        Dimensions of the input and output tensors

    stride_xb1, stride_xn, stride_xp : int
        Strides for indexing into x_ptr
    stride_vb1, stride_vp, stride_vr : int
        Strides for indexing into v_ptr
    stride_sb1, stride_sb2, stride_sr : int
        Strides for indexing into s_ptr
    stride_zb2, stride_zn, stride_zr : int
        Strides for indexing into z_ptr
    stride_yb1, stride_yn, stride_yr : int
        Strides for indexing into y_ptr

    BLOCK_SIZE_N, BLOCK_SIZE_P, BLOCK_SIZE_B1, BLOCK_SIZE_R, BLOCK_SIZE_B2 : tl.constexpr
        Constants defining the block sizes for partitioning the computation
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

    offs_xn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_r = (pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)) % R
    offs_p = tl.arange(0, BLOCK_SIZE_P)
    offs_b2 = tl.arange(0, BLOCK_SIZE_B2)

    accumulator_out = tl.zeros((BLOCK_SIZE_B2, BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)

    for b1 in range(0, B1):
        x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_p[None, :] * stride_xp + b1 * stride_xb1)
        v_ptrs = v_ptr + (offs_p[:, None] * stride_vp + offs_r[None, :] * stride_vr + b1 * stride_vb1)
        s_ptrs = s_ptr + (offs_b2[:, None] * stride_sb2 + offs_r[None, :] * stride_sr + b1 * stride_sb1)
        accumulator_in = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
        s_mask = ((offs_b2[:, None] < B2) & (offs_r[None, :] < R))
        s = tl.load(s_ptrs, mask = s_mask, other=0.0)
        s = tl.expand_dims(s, 1)
        for p in range(0, tl.cdiv(P, BLOCK_SIZE_P)):
            x_mask = (offs_p[None, :] < (P - p * BLOCK_SIZE_P))
            v_mask = (offs_p[:, None] < (P - p * BLOCK_SIZE_P))
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            accumulator_in = tl.dot(x, v, accumulator_in)
            x_ptrs += BLOCK_SIZE_P * stride_xp
            v_ptrs += BLOCK_SIZE_P * stride_vp
        y = accumulator_in.to(tl.bfloat16)

        y = tl.expand_dims(y, 0)
        accumulator_out += s * y

    z = accumulator_out.to(tl.bfloat16)
    offs_zn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_zr = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    z_ptrs = z_ptr + stride_zn * offs_zn[None, :, None] + stride_zr * offs_zr[None, None, :] + stride_zb2 * offs_b2[:, None, None]
    z_mask = ((offs_zn[None, :, None] < N) & (offs_zr[None, None, :] < R) & (offs_b2[:, None, None] < B2))
    tl.store(z_ptrs, z, mask=z_mask)

#-----------------------------------
@triton.jit
def _triton_blast_partial_grouped_persistent_kernel_fp32(
    x_ptr, v_ptr, s_ptr, z_ptr, y_ptr,
    N, P, B1, R, B2,
    stride_xb1, stride_xn, stride_xp,
    stride_vb1, stride_vp,  stride_vr,
    stride_sb1, stride_sb2, stride_sr,
    stride_zb2, stride_zn,  stride_zr,
    stride_yb1, stride_yn, stride_yr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_P: tl.constexpr, BLOCK_SIZE_B1: tl.constexpr, BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_B2: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr, NUM_SMS: tl.constexpr):

    """ 
    Computes batched matrix multiplication between X, V, and S, and produces outputs stored 
    in Y and Z, using FP32 inputs and outputs. Useful for checking correctness. Promotes better 
    data reuse by super-grouping blocks in groups of GROUP_N rows before switching to the next column.
    Applies persistent optimization to reduce overhead of short pipelines and hardware scheduling due
    to large grid sizes.

    Parameters:
    ----------
    x_ptr : tl.tensor
        Pointer to the input matrix X, shape (B1, N, P)
    v_ptr : tl.tensor
        Pointer to the input matrix V, shape (B1, P, R)
    s_ptr : tl.tensor
        Pointer to the scaling matrix S, shape (B1, B2, R)
    z_ptr : tl.tensor
        Pointer to the output matrix Z, shape (B2, N, R)
    y_ptr : tl.tensor
        Pointer to the intermediate output matrix Y, shape (B1, N, R)

    N, P, B1, R, B2 : int
        Dimensions of the input and output tensors

    stride_xb1, stride_xn, stride_xp : int
        Strides for indexing into x_ptr
    stride_vb1, stride_vp, stride_vr : int
        Strides for indexing into v_ptr
    stride_sb1, stride_sb2, stride_sr : int
        Strides for indexing into s_ptr
    stride_zb2, stride_zn, stride_zr : int
        Strides for indexing into z_ptr
    stride_yb1, stride_yn, stride_yr : int
        Strides for indexing into y_ptr

    BLOCK_SIZE_N, BLOCK_SIZE_P, BLOCK_SIZE_B1, BLOCK_SIZE_R, BLOCK_SIZE_B2 : tl.constexpr
        Constants defining the block sizes for partitioning the computation
    """

    start_pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N) 
    num_pid_r = tl.cdiv(R, BLOCK_SIZE_R)
    p_tiles = tl.cdiv(P, BLOCK_SIZE_P)
    num_tiles = num_pid_n * num_pid_r
    
    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1
    tile_id = start_pid - NUM_SMS

    offs_p = tl.arange(0, BLOCK_SIZE_P)
    num_pid_in_group = GROUP_SIZE_N * num_pid_r
    pid_n = 0
    pid_r = 0

    offs_xn = tl.arange(0, BLOCK_SIZE_N)
    offs_r = tl.arange(0, BLOCK_SIZE_R)
    offs_b2 = tl.arange(0, BLOCK_SIZE_B2)
    
    for _ in range(tiles_per_SM):
        tile_id += NUM_SMS
        accumulator_out = tl.zeros((BLOCK_SIZE_B2, BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
        group_id = tile_id // num_pid_in_group
        first_pid_n = group_id * GROUP_SIZE_N
        group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_N)
        pid_n = first_pid_n + (tile_id % group_size_n)
        pid_r = (tile_id % num_pid_in_group) // group_size_n
        start_n = pid_n * BLOCK_SIZE_N
        start_r = pid_r * BLOCK_SIZE_R
        
        offs_xn = (start_n + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_yn = (start_n + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_r = (start_r + tl.arange(0, BLOCK_SIZE_R)) % R
        
        for b1 in range(0, B1):
            accumulator_in = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
            x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_p[None, :] * stride_xp + b1 * stride_xb1)
            v_ptrs = v_ptr + (offs_p[:, None] * stride_vp + offs_r[None, :] * stride_vr + b1 * stride_vb1)
            y_ptrs = y_ptr + (stride_yn * offs_yn[:, None] + stride_yr * offs_r[None, :] + b1 * stride_yb1)
            s_ptrs = s_ptr + (offs_b2[:, None] * stride_sb2 + offs_r[None, :] * stride_sr + b1 * stride_sb1)
            s_mask = ((offs_b2[:, None] < B2) & (offs_r[None, :] < R))
            s = tl.load(s_ptrs, mask=s_mask, other=0.0)
            s = tl.expand_dims(s, 1)
            for p in range(0, tl.cdiv(P, BLOCK_SIZE_P)):
                x_mask = (offs_p[None, :] < P - p * BLOCK_SIZE_P)
                v_mask = (offs_p[:, None] < P - p * BLOCK_SIZE_P)
                x = tl.load(x_ptrs, mask=x_mask, other=0.0)
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
                accumulator_in = tl.dot(x, v, accumulator_in, allow_tf32=False)
                x_ptrs += stride_xp * BLOCK_SIZE_P 
                v_ptrs += stride_vp * BLOCK_SIZE_P 
            y_mask = ((offs_yn[:, None] < N) & (offs_r[None, :] < R))
            y = accumulator_in
            tl.store(y_ptrs, y, mask=y_mask)

            y = tl.expand_dims(y, 0)
            accumulator_out += s * y

        z = accumulator_out
        offs_zn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_zr = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
        z_ptrs = z_ptr + stride_zn * offs_zn[None, :, None] + stride_zr * offs_zr[None, None, :] + stride_zb2 * offs_b2[:, None, None]
        z_mask = ((offs_zn[None, :, None] < N) & (offs_zr[None, None, :] < R) & (offs_b2[:, None, None] < B2))
        tl.store(z_ptrs, z, mask=z_mask)

@triton.autotune(configs=_get_triton_blast_partial_grouped_persistent_kernel_autotune_config(), key=['N', 'P', 'R', 'B1', 'B2'])
@triton.jit
def _triton_blast_partial_grouped_persistent_kernel_fp16(
    x_ptr, v_ptr, s_ptr, z_ptr,
    N, P, B1, R, B2,
    stride_xb1, stride_xn, stride_xp,
    stride_vb1, stride_vp,  stride_vr,
    stride_sb1, stride_sb2, stride_sr,
    stride_zb2, stride_zn,  stride_zr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_P: tl.constexpr, BLOCK_SIZE_B1: tl.constexpr, BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_B2: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr, NUM_SMS: tl.constexpr):

    """ 
    Computes batched matrix multiplication between X, V, and S, and produces outputs stored 
    in Y and Z, using FP16 inputs and outputs. Useful for performance benchmarking. Promotes better 
    data reuse by super-grouping blocks in groups of GROUP_N rows before switching to the next column.
    Applies persistent optimization to reduce overhead of short pipelines and hardware scheduling due
    to large grid sizes.

    Parameters:
    ----------
    x_ptr : tl.tensor
        Pointer to the input matrix X, shape (B1, N, P)
    v_ptr : tl.tensor
        Pointer to the input matrix V, shape (B1, P, R)
    s_ptr : tl.tensor
        Pointer to the scaling matrix S, shape (B1, B2, R)
    z_ptr : tl.tensor
        Pointer to the output matrix Z, shape (B2, N, R)
    y_ptr : tl.tensor
        Pointer to the intermediate output matrix Y, shape (B1, N, R)

    N, P, B1, R, B2 : int
        Dimensions of the input and output tensors

    stride_xb1, stride_xn, stride_xp : int
        Strides for indexing into x_ptr
    stride_vb1, stride_vp, stride_vr : int
        Strides for indexing into v_ptr
    stride_sb1, stride_sb2, stride_sr : int
        Strides for indexing into s_ptr
    stride_zb2, stride_zn, stride_zr : int
        Strides for indexing into z_ptr
    stride_yb1, stride_yn, stride_yr : int
        Strides for indexing into y_ptr

    BLOCK_SIZE_N, BLOCK_SIZE_P, BLOCK_SIZE_B1, BLOCK_SIZE_R, BLOCK_SIZE_B2 : tl.constexpr
        Constants defining the block sizes for partitioning the computation
    """

    start_pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N) 
    num_pid_r = tl.cdiv(R, BLOCK_SIZE_R)
    p_tiles = tl.cdiv(P, BLOCK_SIZE_P)
    num_tiles = num_pid_n * num_pid_r
    
    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1
    tile_id = start_pid - NUM_SMS

    offs_p = tl.arange(0, BLOCK_SIZE_P)
    num_pid_in_group = GROUP_SIZE_N * num_pid_r
    pid_n = 0
    pid_r = 0

    offs_xn = tl.arange(0, BLOCK_SIZE_N)
    offs_r = tl.arange(0, BLOCK_SIZE_R)
    offs_b2 = tl.arange(0, BLOCK_SIZE_B2)
    
    for _ in range(tiles_per_SM):
        tile_id += NUM_SMS
        accumulator_out = tl.zeros((BLOCK_SIZE_B2, BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
        group_id = tile_id // num_pid_in_group
        first_pid_n = group_id * GROUP_SIZE_N
        group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_N)
        pid_n = first_pid_n + (tile_id % group_size_n)
        pid_r = (tile_id % num_pid_in_group) // group_size_n
        start_n = pid_n * BLOCK_SIZE_N
        start_r = pid_r * BLOCK_SIZE_R
        
        offs_xn = (start_n + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_r = (start_r + tl.arange(0, BLOCK_SIZE_R)) % R
        
        for b1 in range(0, B1):
            accumulator_in = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
            x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_p[None, :] * stride_xp + b1 * stride_xb1)
            v_ptrs = v_ptr + (offs_p[:, None] * stride_vp + offs_r[None, :] * stride_vr + b1 * stride_vb1)
            s_ptrs = s_ptr + (offs_b2[:, None] * stride_sb2 + offs_r[None, :] * stride_sr + b1 * stride_sb1)
            s_mask = ((offs_b2[:, None] < B2) & (offs_r[None, :] < R))
            s = tl.load(s_ptrs, mask=s_mask, other=0.0)
            s = tl.expand_dims(s, 1)
            for p in range(0, tl.cdiv(P, BLOCK_SIZE_P)):
                x_mask = (offs_p[None, :] < P - p * BLOCK_SIZE_P)
                v_mask = (offs_p[:, None] < P - p * BLOCK_SIZE_P)
                x = tl.load(x_ptrs, mask=x_mask, other=0.0)
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
                accumulator_in = tl.dot(x, v, accumulator_in)
                x_ptrs += stride_xp * BLOCK_SIZE_P 
                v_ptrs += stride_vp * BLOCK_SIZE_P 
            y = accumulator_in.to(tl.bfloat16)

            y = tl.expand_dims(y, 0)
            accumulator_out += s * y

        z = accumulator_out.to(tl.bfloat16)
        offs_zn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_zr = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
        z_ptrs = z_ptr + stride_zn * offs_zn[None, :, None] + stride_zr * offs_zr[None, None, :] + stride_zb2 * offs_b2[:, None, None]
        z_mask = ((offs_zn[None, :, None] < N) & (offs_zr[None, None, :] < R) & (offs_b2[:, None, None] < B2))
        tl.store(z_ptrs, z, mask=z_mask)

@triton.jit
def _triton_blast_partial_grouped_persistent_kernel_fp16_no_autotune(
    x_ptr, v_ptr, s_ptr, z_ptr,
    N, P, B1, R, B2,
    stride_xb1, stride_xn, stride_xp,
    stride_vb1, stride_vp,  stride_vr,
    stride_sb1, stride_sb2, stride_sr,
    stride_zb2, stride_zn,  stride_zr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_P: tl.constexpr, BLOCK_SIZE_B1: tl.constexpr, BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_B2: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr, NUM_SMS: tl.constexpr):

    """ 
    Computes batched matrix multiplication between X, V, and S, and produces outputs stored 
    in Y and Z, using FP16 inputs and outputs. Useful for performance benchmarking. Promotes better 
    data reuse by super-grouping blocks in groups of GROUP_N rows before switching to the next column.
    Applies persistent optimization to reduce overhead of short pipelines and hardware scheduling due
    to large grid sizes.

    Parameters:
    ----------
    x_ptr : tl.tensor
        Pointer to the input matrix X, shape (B1, N, P)
    v_ptr : tl.tensor
        Pointer to the input matrix V, shape (B1, P, R)
    s_ptr : tl.tensor
        Pointer to the scaling matrix S, shape (B1, B2, R)
    z_ptr : tl.tensor
        Pointer to the output matrix Z, shape (B2, N, R)
    y_ptr : tl.tensor
        Pointer to the intermediate output matrix Y, shape (B1, N, R)

    N, P, B1, R, B2 : int
        Dimensions of the input and output tensors

    stride_xb1, stride_xn, stride_xp : int
        Strides for indexing into x_ptr
    stride_vb1, stride_vp, stride_vr : int
        Strides for indexing into v_ptr
    stride_sb1, stride_sb2, stride_sr : int
        Strides for indexing into s_ptr
    stride_zb2, stride_zn, stride_zr : int
        Strides for indexing into z_ptr
    stride_yb1, stride_yn, stride_yr : int
        Strides for indexing into y_ptr

    BLOCK_SIZE_N, BLOCK_SIZE_P, BLOCK_SIZE_B1, BLOCK_SIZE_R, BLOCK_SIZE_B2 : tl.constexpr
        Constants defining the block sizes for partitioning the computation
    """

    start_pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N) 
    num_pid_r = tl.cdiv(R, BLOCK_SIZE_R)
    p_tiles = tl.cdiv(P, BLOCK_SIZE_P)
    num_tiles = num_pid_n * num_pid_r
    
    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1
    tile_id = start_pid - NUM_SMS

    offs_p = tl.arange(0, BLOCK_SIZE_P)
    num_pid_in_group = GROUP_SIZE_N * num_pid_r
    pid_n = 0
    pid_r = 0

    offs_xn = tl.arange(0, BLOCK_SIZE_N)
    offs_r = tl.arange(0, BLOCK_SIZE_R)
    offs_b2 = tl.arange(0, BLOCK_SIZE_B2)
    
    for _ in range(tiles_per_SM):
        tile_id += NUM_SMS
        accumulator_out = tl.zeros((BLOCK_SIZE_B2, BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
        group_id = tile_id // num_pid_in_group
        first_pid_n = group_id * GROUP_SIZE_N
        group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_N)
        pid_n = first_pid_n + (tile_id % group_size_n)
        pid_r = (tile_id % num_pid_in_group) // group_size_n
        start_n = pid_n * BLOCK_SIZE_N
        start_r = pid_r * BLOCK_SIZE_R
        
        offs_xn = (start_n + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_r = (start_r + tl.arange(0, BLOCK_SIZE_R)) % R
        
        for b1 in range(0, B1):
            accumulator_in = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
            x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_p[None, :] * stride_xp + b1 * stride_xb1)
            v_ptrs = v_ptr + (offs_p[:, None] * stride_vp + offs_r[None, :] * stride_vr + b1 * stride_vb1)
            s_ptrs = s_ptr + (offs_b2[:, None] * stride_sb2 + offs_r[None, :] * stride_sr + b1 * stride_sb1)
            s_mask = ((offs_b2[:, None] < B2) & (offs_r[None, :] < R))
            s = tl.load(s_ptrs, mask=s_mask, other=0.0)
            s = tl.expand_dims(s, 1)
            for p in range(0, tl.cdiv(P, BLOCK_SIZE_P)):
                x_mask = (offs_p[None, :] < P - p * BLOCK_SIZE_P)
                v_mask = (offs_p[:, None] < P - p * BLOCK_SIZE_P)
                x = tl.load(x_ptrs, mask=x_mask, other=0.0)
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
                accumulator_in = tl.dot(x, v, accumulator_in)
                x_ptrs += stride_xp * BLOCK_SIZE_P 
                v_ptrs += stride_vp * BLOCK_SIZE_P 
            y = accumulator_in.to(tl.bfloat16)

            y = tl.expand_dims(y, 0)
            accumulator_out += s * y

        z = accumulator_out.to(tl.bfloat16)
        offs_zn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_zr = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
        z_ptrs = z_ptr + stride_zn * offs_zn[None, :, None] + stride_zr * offs_zr[None, None, :] + stride_zb2 * offs_b2[:, None, None]
        z_mask = ((offs_zn[None, :, None] < N) & (offs_zr[None, None, :] < R) & (offs_b2[:, None, None] < B2))
        tl.store(z_ptrs, z, mask=z_mask)

#-----------------------------------
@triton.jit
def _triton_blast_full_kernel_fp32(
    x_ptr, v_ptr, s_ptr, u_ptr, o_ptr, z_ptr, y_ptr,
    N, P, B1, R, B2, Q,
    stride_xb1, stride_xn, stride_xp,
    stride_vb1, stride_vp, stride_vr,
    stride_sb1, stride_sb2, stride_sr,
    stride_ub2, stride_ur, stride_uq,
    stride_ob2, stride_on, stride_oq,
    stride_zb2, stride_zn, stride_zr,
    stride_yb1, stride_yn, stride_yr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_P: tl.constexpr, BLOCK_SIZE_B1: tl.constexpr, 
    BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_B2: tl.constexpr, BLOCK_SIZE_Q: tl.constexpr):

    """
    Computes batched matrix multiplication between X, V, S, and U and produces outputs stored 
    in O, Y, and Z, using FP32 inputs and outputs. Useful for checking correctness. Currently
    broken because Triton does not support 3D tensor tl.dot 

    The computation consists of two stages:
    
    1. X * V * S = Y * S = Z:
       - Iterates over blocks in B1, processing input matrices x_ptr and v_ptr using matrix multiplication
       - Applies scaling factors from s_ptr and accumulates the result in accumulator_out
       - Stores intermediate results in y_ptr and final accumulated results in z_ptr

    2. Z * U = O
       - Loads u_ptr, applies additional matrix multiplication with accumulator_out
       - Stores the final result in o_ptr, producing the complete output

    Parameters:
    ----------
    x_ptr : tl.tensor
        Pointer to the input matrix X, shape (B1, N, P)
    v_ptr : tl.tensor
        Pointer to the input matrix V, shape (B1, P, R)
    s_ptr : tl.tensor
        Pointer to the scaling matrix S, shape (B1, B2, R)
    u_ptr : tl.tensor
        Pointer to the transformation matrix U, shape (B2, R, Q)
    o_ptr : tl.tensor
        Pointer to the final output matrix O, shape (B2, N, Q)
    z_ptr : tl.tensor
        Pointer to the intermediate accumulated matrix Z, shape (B2, N, R)
    y_ptr : tl.tensor
        Pointer to the intermediate output matrix Y, shape (B1, N, R)

    N, P, B1, R, B2, Q : int
        Dimensions of the input and output tensors

    stride_xb1, stride_xn, stride_xp : int
        Strides for indexing into x_ptr
    stride_vb1, stride_vp, stride_vr : int
        Strides for indexing into v_ptr
    stride_sb1, stride_sb2, stride_sr : int
        Strides for indexing into s_ptr
    stride_ub2, stride_ur, stride_uq : int
        Strides for indexing into u_ptr
    stride_ob2, stride_on, stride_oq : int
        Strides for indexing into o_ptr
    stride_zb2, stride_zn, stride_zr : int
        Strides for indexing into z_ptr
    stride_yb1, stride_yn, stride_yr : int
        Strides for indexing into y_ptr

    BLOCK_SIZE_N, BLOCK_SIZE_P, BLOCK_SIZE_B1, BLOCK_SIZE_R, BLOCK_SIZE_B2, BLOCK_SIZE_Q : tl.constexpr
        Constants defining the block sizes for partitioning the computation
    """

    pid_n = tl.program_id(axis=0)
    offs_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_p = tl.arange(0, BLOCK_SIZE_P)
    offs_r = tl.arange(0, BLOCK_SIZE_R)
    offs_b2 = tl.arange(0, BLOCK_SIZE_B2)

    accumulator_out = tl.zeros((BLOCK_SIZE_B2, BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
    for b1 in range(0, B1):
        x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_p[None, :] * stride_xp + b1 * stride_xb1)
        v_ptrs = v_ptr + (offs_p[:, None] * stride_vp + offs_r[None, :] * stride_vr + b1 * stride_vb1)
        s_ptrs = s_ptr + (offs_b2[:, None] * stride_sb2 + offs_r[None, :] * stride_sr + b1 * stride_sb1)
        y_ptrs = y_ptr + (stride_yn * offs_yn[:, None] + stride_yr * offs_r[None, :] + b1 * stride_yb1)
        accumulator_in = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
        s_mask = ((offs_b2[:, None] < B2) & (offs_r[None, :] < R))
        s = tl.load(s_ptrs, mask = s_mask, other=0.0)
        s = tl.expand_dims(s, 1)
        for p in range(0, tl.cdiv(P, BLOCK_SIZE_P)):
            x_mask = ((offs_p[None, :] < (P - p * BLOCK_SIZE_P)) & (offs_xn[:, None] < N))
            v_mask = ((offs_p[:, None] < (P - p * BLOCK_SIZE_P)) & (offs_r[None, :] < R))
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            accumulator_in = tl.dot(x, v, accumulator_in, allow_tf32=False)
            x_ptrs += BLOCK_SIZE_P * stride_xp
            v_ptrs += BLOCK_SIZE_P * stride_vp

        y = accumulator_in
        y_mask = ((offs_yn[:, None] < N) & (offs_r[None, :] < R))
        tl.store(y_ptrs, y, mask=y_mask)

        y = tl.expand_dims(y, 0)
        accumulator_out += s * y

    z = accumulator_out
    offs_zn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    z_ptrs = z_ptr + (stride_zn * offs_zn[None, :, None] + stride_zr * offs_r[None, None, :] + stride_zb2 * offs_b2[:, None, None])
    z_mask = ((offs_zn[None, :, None] < N) & (offs_r[None, None, :] < R) & (offs_b2[:, None, None] < B2))
    tl.store(z_ptrs, z, mask=z_mask)

    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_q = tl.arange(0, BLOCK_SIZE_Q)
    o_ptrs = o_ptr + (stride_on * offs_on[None, :, None] + stride_oq * offs_q[None, None, :] + stride_ob2 * offs_b2[:, None, None])
    u_ptrs = u_ptr + (offs_b2[:, None, None] * stride_ub2 + offs_r[None, :, None] * stride_ur + offs_q[None, None,:] * stride_uq)
    for q in range(0, tl.cdiv(Q, BLOCK_SIZE_Q)):
        u_mask = ((offs_r[None, :, None] < R) & (offs_q[None, None, :] < (Q - q * BLOCK_SIZE_Q)) & (offs_b2[:, None, None] < B2))
        o_mask = ((offs_on[None, :, None] < N) & (offs_q[None, None, :] < (Q - q * BLOCK_SIZE_Q)) & (offs_b2[:, None, None] < B2))
        u = tl.load(u_ptrs, mask=u_mask, other=0.0)
        o = tl.dot(z, u, allow_tf32=False)
        tl.store(o_ptrs, o, mask=o_mask)
        o_ptrs += BLOCK_SIZE_Q * stride_oq
        u_ptrs += BLOCK_SIZE_Q * stride_uq

@triton.autotune(configs=_get_triton_blast_full_kernel_autotune_config(), key=['N', 'P', 'R', 'Q', 'B1', 'B2'])
@triton.jit
def _triton_blast_full_kernel_fp16(
    x_ptr, v_ptr, s_ptr, u_ptr, o_ptr,
    N, P, B1, R, B2, Q,
    stride_xb1, stride_xn, stride_xp,
    stride_vb1, stride_vp, stride_vr,
    stride_sb1, stride_sb2, stride_sr,
    stride_ub2, stride_ur, stride_uq,
    stride_ob2, stride_on, stride_oq,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_P: tl.constexpr, BLOCK_SIZE_B1: tl.constexpr, 
    BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_B2: tl.constexpr, BLOCK_SIZE_Q: tl.constexpr):

    """
    Computes batched matrix multiplication between X, V, S, and U and produces outputs stored 
    in O, Y, and Z, using FP16 inputs and outputs. Useful for performance benchmarking. Currently
    broken because Triton does not support 3D tensor tl.dot 

    The computation consists of two stages:
    
    1. X * V * S = Y * S = Z:
       - Iterates over blocks in B1, processing input matrices x_ptr and v_ptr using matrix multiplication
       - Applies scaling factors from s_ptr and accumulates the result in accumulator_out
       - Stores intermediate results in y_ptr and final accumulated results in z_ptr

    2. Z * U = O
       - Loads u_ptr, applies additional matrix multiplication with accumulator_out
       - Stores the final result in o_ptr, producing the complete output

    Parameters:
    ----------
    x_ptr : tl.tensor
        Pointer to the input matrix X, shape (B1, N, P)
    v_ptr : tl.tensor
        Pointer to the input matrix V, shape (B1, P, R)
    s_ptr : tl.tensor
        Pointer to the scaling matrix S, shape (B1, B2, R)
    u_ptr : tl.tensor
        Pointer to the transformation matrix U, shape (B2, R, Q)
    o_ptr : tl.tensor
        Pointer to the final output matrix O, shape (B2, N, Q)
    z_ptr : tl.tensor
        Pointer to the intermediate accumulated matrix Z, shape (B2, N, R)
    y_ptr : tl.tensor
        Pointer to the intermediate output matrix Y, shape (B1, N, R)

    N, P, B1, R, B2, Q : int
        Dimensions of the input and output tensors

    stride_xb1, stride_xn, stride_xp : int
        Strides for indexing into x_ptr
    stride_vb1, stride_vp, stride_vr : int
        Strides for indexing into v_ptr
    stride_sb1, stride_sb2, stride_sr : int
        Strides for indexing into s_ptr
    stride_ub2, stride_ur, stride_uq : int
        Strides for indexing into u_ptr
    stride_ob2, stride_on, stride_oq : int
        Strides for indexing into o_ptr
    stride_zb2, stride_zn, stride_zr : int
        Strides for indexing into z_ptr
    stride_yb1, stride_yn, stride_yr : int
        Strides for indexing into y_ptr

    BLOCK_SIZE_N, BLOCK_SIZE_P, BLOCK_SIZE_B1, BLOCK_SIZE_R, BLOCK_SIZE_B2, BLOCK_SIZE_Q : tl.constexpr
        Constants defining the block sizes for partitioning the computation
    """

    pid_n = tl.program_id(axis=0)
    offs_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_p = tl.arange(0, BLOCK_SIZE_P)
    offs_r = tl.arange(0, BLOCK_SIZE_R)
    offs_b2 = tl.arange(0, BLOCK_SIZE_B2)

    accumulator_out = tl.zeros((BLOCK_SIZE_B2, BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
    for b1 in range(0, B1):
        x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_p[None, :] * stride_xp + b1 * stride_xb1)
        v_ptrs = v_ptr + (offs_p[:, None] * stride_vp + offs_r[None, :] * stride_vr + b1 * stride_vb1)
        s_ptrs = s_ptr + (offs_b2[:, None] * stride_sb2 + offs_r[None, :] * stride_sr + b1 * stride_sb1)
        accumulator_in = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
        s_mask = ((offs_b2[:, None] < B2) & (offs_r[None, :] < R))
        s = tl.load(s_ptrs, mask = s_mask, other=0.0)
        s = tl.expand_dims(s, 1)
        for p in range(0, tl.cdiv(P, BLOCK_SIZE_P)):
            x_mask = ((offs_p[None, :] < (P - p * BLOCK_SIZE_P)) & (offs_xn[:, None] < N))
            v_mask = ((offs_p[:, None] < (P - p * BLOCK_SIZE_P)) & (offs_r[None, :] < R))
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            accumulator_in = tl.dot(x, v, accumulator_in)
            x_ptrs += BLOCK_SIZE_P * stride_xp
            v_ptrs += BLOCK_SIZE_P * stride_vp
        y = accumulator_in.to(tl.bfloat16)

        y = tl.expand_dims(y, 0)
        accumulator_out += s * y

    z = accumulator_out.to(tl.bfloat16)

    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_q = tl.arange(0, BLOCK_SIZE_Q)
    o_ptrs = o_ptr + (stride_on * offs_on[None, :, None] + stride_oq * offs_q[None, None, :] + stride_ob2 * offs_b2[:, None, None])
    u_ptrs = u_ptr + (offs_b2[:, None, None] * stride_ub2 + offs_r[None, :, None] * stride_ur + offs_q[None, None,:] * stride_uq)
    for q in range(0, tl.cdiv(Q, BLOCK_SIZE_Q)):
        u_mask = ((offs_r[None, :, None] < R) & (offs_q[None, None, :] < (Q - q * BLOCK_SIZE_Q)) & (offs_b2[:, None, None] < B2))
        o_mask = ((offs_on[None, :, None] < N) & (offs_q[None, None, :] < (Q - q * BLOCK_SIZE_Q)) & (offs_b2[:, None, None] < B2))
        u = tl.load(u_ptrs, mask=u_mask, other=0.0)
        o = tl.dot(z, u)
        o = o.to(tl.bfloat16)
        tl.store(o_ptrs, o, mask=o_mask)
        o_ptrs += BLOCK_SIZE_Q * stride_oq
        u_ptrs += BLOCK_SIZE_Q * stride_uq

@triton.jit
def _triton_blast_full_kernel_fp16_no_autotune(
    x_ptr, v_ptr, s_ptr, u_ptr, o_ptr,
    N, P, B1, R, B2, Q,
    stride_xb1, stride_xn, stride_xp,
    stride_vb1, stride_vp, stride_vr,
    stride_sb1, stride_sb2, stride_sr,
    stride_ub2, stride_ur, stride_uq,
    stride_ob2, stride_on, stride_oq,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_P: tl.constexpr, BLOCK_SIZE_B1: tl.constexpr, 
    BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_B2: tl.constexpr, BLOCK_SIZE_Q: tl.constexpr):

    """
    Computes batched matrix multiplication between X, V, S, and U and produces outputs stored 
    in O, Y, and Z, using FP16 inputs and outputs. Useful for performance benchmarking. Currently
    broken because Triton does not support 3D tensor tl.dot 

    The computation consists of two stages:
    
    1. X * V * S = Y * S = Z:
       - Iterates over blocks in B1, processing input matrices x_ptr and v_ptr using matrix multiplication
       - Applies scaling factors from s_ptr and accumulates the result in accumulator_out
       - Stores intermediate results in y_ptr and final accumulated results in z_ptr

    2. Z * U = O
       - Loads u_ptr, applies additional matrix multiplication with accumulator_out
       - Stores the final result in o_ptr, producing the complete output

    Parameters:
    ----------
    x_ptr : tl.tensor
        Pointer to the input matrix X, shape (B1, N, P)
    v_ptr : tl.tensor
        Pointer to the input matrix V, shape (B1, P, R)
    s_ptr : tl.tensor
        Pointer to the scaling matrix S, shape (B1, B2, R)
    u_ptr : tl.tensor
        Pointer to the transformation matrix U, shape (B2, R, Q)
    o_ptr : tl.tensor
        Pointer to the final output matrix O, shape (B2, N, Q)
    z_ptr : tl.tensor
        Pointer to the intermediate accumulated matrix Z, shape (B2, N, R)
    y_ptr : tl.tensor
        Pointer to the intermediate output matrix Y, shape (B1, N, R)

    N, P, B1, R, B2, Q : int
        Dimensions of the input and output tensors

    stride_xb1, stride_xn, stride_xp : int
        Strides for indexing into x_ptr
    stride_vb1, stride_vp, stride_vr : int
        Strides for indexing into v_ptr
    stride_sb1, stride_sb2, stride_sr : int
        Strides for indexing into s_ptr
    stride_ub2, stride_ur, stride_uq : int
        Strides for indexing into u_ptr
    stride_ob2, stride_on, stride_oq : int
        Strides for indexing into o_ptr
    stride_zb2, stride_zn, stride_zr : int
        Strides for indexing into z_ptr
    stride_yb1, stride_yn, stride_yr : int
        Strides for indexing into y_ptr

    BLOCK_SIZE_N, BLOCK_SIZE_P, BLOCK_SIZE_B1, BLOCK_SIZE_R, BLOCK_SIZE_B2, BLOCK_SIZE_Q : tl.constexpr
        Constants defining the block sizes for partitioning the computation
    """

    pid_n = tl.program_id(axis=0)
    offs_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_p = tl.arange(0, BLOCK_SIZE_P)
    offs_r = tl.arange(0, BLOCK_SIZE_R)
    offs_b2 = tl.arange(0, BLOCK_SIZE_B2)

    accumulator_out = tl.zeros((BLOCK_SIZE_B2, BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
    for b1 in range(0, B1):
        x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_p[None, :] * stride_xp + b1 * stride_xb1)
        v_ptrs = v_ptr + (offs_p[:, None] * stride_vp + offs_r[None, :] * stride_vr + b1 * stride_vb1)
        s_ptrs = s_ptr + (offs_b2[:, None] * stride_sb2 + offs_r[None, :] * stride_sr + b1 * stride_sb1)
        accumulator_in = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
        s_mask = ((offs_b2[:, None] < B2) & (offs_r[None, :] < R))
        s = tl.load(s_ptrs, mask = s_mask, other=0.0)
        s = tl.expand_dims(s, 1)
        for p in range(0, tl.cdiv(P, BLOCK_SIZE_P)):
            x_mask = ((offs_p[None, :] < (P - p * BLOCK_SIZE_P)) & (offs_xn[:, None] < N))
            v_mask = ((offs_p[:, None] < (P - p * BLOCK_SIZE_P)) & (offs_r[None, :] < R))
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            accumulator_in = tl.dot(x, v, accumulator_in)
            x_ptrs += BLOCK_SIZE_P * stride_xp
            v_ptrs += BLOCK_SIZE_P * stride_vp
        y = accumulator_in.to(tl.bfloat16)

        y = tl.expand_dims(y, 0)
        accumulator_out += s * y

    z = accumulator_out.to(tl.bfloat16)

    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_q = tl.arange(0, BLOCK_SIZE_Q)
    o_ptrs = o_ptr + (stride_on * offs_on[None, :, None] + stride_oq * offs_q[None, None, :] + stride_ob2 * offs_b2[:, None, None])
    u_ptrs = u_ptr + (offs_b2[:, None, None] * stride_ub2 + offs_r[None, :, None] * stride_ur + offs_q[None, None,:] * stride_uq)
    for q in range(0, tl.cdiv(Q, BLOCK_SIZE_Q)):
        u_mask = ((offs_r[None, :, None] < R) & (offs_q[None, None, :] < (Q - q * BLOCK_SIZE_Q)) & (offs_b2[:, None, None] < B2))
        o_mask = ((offs_on[None, :, None] < N) & (offs_q[None, None, :] < (Q - q * BLOCK_SIZE_Q)) & (offs_b2[:, None, None] < B2))
        u = tl.load(u_ptrs, mask=u_mask, other=0.0)
        o = tl.dot(z, u)
        o = o.to(tl.bfloat16)
        tl.store(o_ptrs, o, mask=o_mask)
        o_ptrs += BLOCK_SIZE_Q * stride_oq
        u_ptrs += BLOCK_SIZE_Q * stride_uq

#-----------------------------------
@triton.jit
def _triton_blast_bmm_xv_kernel_fp32(
    x_ptr, v_ptr, y_ptr,
    N, P, B1, R,
    stride_xb1, stride_xn, stride_xp,
    stride_vb1, stride_vp, stride_vr,
    stride_yr, stride_yb1, stride_yn,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_P: tl.constexpr, BLOCK_SIZE_R: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr):

    """
    Triton kernel for performing batched matrix multiplication: Y[b1, r, n] = X[b1, n, p] @ V[b1, p, r]

    This kernel computes the batched matrix multiplication between two input tensors X and V, storing the
    result in Y. The computation is done in float32 accumulation.
    
    Parameters:
    -----------
    x_ptr : tl.pointer
        Pointer to the input tensor X of shape (B1, N, P), in FP32.
    v_ptr : tl.pointer
        Pointer to the input tensor V of shape (B1, P, R), in FP32.
    y_ptr : tl.pointer
        Pointer to the output tensor Y of shape (B1, R, N), in FP32.
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

    x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_p[None, :] * stride_xp + b1 * stride_xb1)
    v_ptrs = v_ptr + (offs_p[:, None] * stride_vp + offs_r[None, :] * stride_vr + b1 * stride_vb1)
    accumulator_in = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
    for p in range(0, tl.cdiv(P, BLOCK_SIZE_P)):
        x_mask = (offs_p[None, :] < (P - p * BLOCK_SIZE_P))
        v_mask = (offs_p[:, None] < (P - p * BLOCK_SIZE_P))
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        accumulator_in = tl.dot(x, v, accumulator_in, allow_tf32=False)
        x_ptrs += BLOCK_SIZE_P * stride_xp
        v_ptrs += BLOCK_SIZE_P * stride_vp
    
    y = accumulator_in.trans()

    offs_yn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_yr = (pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R))
    y_ptrs = y_ptr + offs_yn[None, :] * stride_yn + offs_yr[:, None] * stride_yr + b1 * stride_yb1
    y_mask = (offs_yn[None, :] < N) & (offs_yr[:, None] < R)
    tl.store(y_ptrs, y, mask=y_mask)

@triton.autotune(configs=_get_triton_blast_bmm_xv_kernel_autotune_config(), key=['N', 'P', 'R', 'B1'])
@triton.jit
def _triton_blast_bmm_xv_kernel_fp16(
    x_ptr, v_ptr, y_ptr,
    N, P, B1, R,
    stride_xb1, stride_xn, stride_xp,
    stride_vb1, stride_vp, stride_vr,
    stride_yr, stride_yb1, stride_yn,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_P: tl.constexpr, BLOCK_SIZE_R: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr):

    """
    Triton kernel for performing batched matrix multiplication: Y[b1, r, n] = X[b1, n, p] @ V[b1, p, r]

    This kernel computes the batched matrix multiplication between two input tensors X and V, storing the
    result in Y. The computation is done in float32 accumulation and the result is downcast to bfloat16.
    
    Parameters:
    -----------
    x_ptr : tl.pointer
        Pointer to the input tensor X of shape (B1, N, P), in BF16.
    v_ptr : tl.pointer
        Pointer to the input tensor V of shape (B1, P, R), in BF16.
    y_ptr : tl.pointer
        Pointer to the output tensor Y of shape (B1, R, N), in BF16.
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

    x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_p[None, :] * stride_xp + b1 * stride_xb1)
    v_ptrs = v_ptr + (offs_p[:, None] * stride_vp + offs_r[None, :] * stride_vr + b1 * stride_vb1)
    accumulator_in = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
    for p in range(0, tl.cdiv(P, BLOCK_SIZE_P)):
        x_mask = (offs_p[None, :] < (P - p * BLOCK_SIZE_P))
        v_mask = (offs_p[:, None] < (P - p * BLOCK_SIZE_P))
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        accumulator_in = tl.dot(x, v, accumulator_in)
        x_ptrs += BLOCK_SIZE_P * stride_xp
        v_ptrs += BLOCK_SIZE_P * stride_vp
    
    y = accumulator_in.to(tl.bfloat16).trans()

    offs_yn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_yr = (pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R))
    y_ptrs = y_ptr + offs_yn[None, :] * stride_yn + offs_yr[:, None] * stride_yr + b1 * stride_yb1
    y_mask = (offs_yn[None, :] < N) & (offs_yr[:, None] < R)
    tl.store(y_ptrs, y, mask=y_mask)

@triton.jit
def _triton_blast_bmm_xv_kernel_fp16_no_autotune(
    x_ptr, v_ptr, y_ptr,
    N, P, B1, R,
    stride_xb1, stride_xn, stride_xp,
    stride_vb1, stride_vp, stride_vr,
    stride_yr, stride_yb1, stride_yn,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_P: tl.constexpr, BLOCK_SIZE_R: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr):

    """
    Triton kernel for performing batched matrix multiplication: Y[b1, r, n] = X[b1, n, p] @ V[b1, p, r]

    This kernel computes the batched matrix multiplication between two input tensors X and V, storing the
    result in Y. The computation is done in float32 accumulation and the result is downcast to bfloat16.
    
    Parameters:
    -----------
    x_ptr : tl.pointer
        Pointer to the input tensor X of shape (B1, N, P), in BF16.
    v_ptr : tl.pointer
        Pointer to the input tensor V of shape (B1, P, R), in BF16.
    y_ptr : tl.pointer
        Pointer to the output tensor Y of shape (B1, R, N), in BF16.
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

    x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_p[None, :] * stride_xp + b1 * stride_xb1)
    v_ptrs = v_ptr + (offs_p[:, None] * stride_vp + offs_r[None, :] * stride_vr + b1 * stride_vb1)
    accumulator_in = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)
    for p in range(0, tl.cdiv(P, BLOCK_SIZE_P)):
        x_mask = (offs_p[None, :] < (P - p * BLOCK_SIZE_P))
        v_mask = (offs_p[:, None] < (P - p * BLOCK_SIZE_P))
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        accumulator_in = tl.dot(x, v, accumulator_in)
        x_ptrs += BLOCK_SIZE_P * stride_xp
        v_ptrs += BLOCK_SIZE_P * stride_vp
    
    y = accumulator_in.to(tl.bfloat16).trans()

    offs_yn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_yr = (pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R))
    y_ptrs = y_ptr + offs_yn[None, :] * stride_yn + offs_yr[:, None] * stride_yr + b1 * stride_yb1
    y_mask = (offs_yn[None, :] < N) & (offs_yr[:, None] < R)
    tl.store(y_ptrs, y, mask=y_mask)

#-----------------------------------
@triton.jit
def _triton_blast_bmm_sxv_kernel_fp32(
    y_ptr, s_ptr, z_ptr,
    N, B1, B2, R,
    stride_sr, stride_sb2, stride_sb1,
    stride_yr, stride_yb1, stride_yn,
    stride_zb2, stride_zr, stride_zn,
    BLOCK_SIZE_B2: tl.constexpr, BLOCK_SIZE_B1: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):

    """
    Triton kernel for batched matrix multiplication: Z[b2, r, n] = S[r, b2, b1] @ Y[r, b1, n]

    This kernel computes a batched matrix multiplication where each output tile Z[b2, r, n]
    is the result of multiplying a tensor S[r, b2, b1] with a tensor Y[r, b1, n].

    Parameters:
    -----------
    y_ptr : tl.pointer
        Pointer to the input tensor Y of shape (R, B1, N), stored in FP32.
    s_ptr : tl.pointer
        Pointer to the input tensor S of shape (R, B2, B1), stored in FP32.
    z_ptr : tl.pointer
        Pointer to the output tensor Z of shape (B2, R, N), stored in FP32.

    Notes:
    ------
    - Assumes B1, B2 >= 16, power of two, and not too large (~ <= 128)
    """

    pid_n = tl.program_id(axis=0)
    pid_r = tl.program_id(axis=1)

    offs_yn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_b2 = (tl.arange(0, BLOCK_SIZE_B2)) % B2
    offs_b1 = tl.arange(0, BLOCK_SIZE_B1)

    y_ptrs = y_ptr + (offs_yn[None, :] * stride_yn + offs_b1[:, None] * stride_yb1 + pid_r * stride_yr)
    s_ptrs = s_ptr + (offs_b2[:, None] * stride_sb2 + offs_b1[None, :] * stride_sb1 + pid_r * stride_sr)

    accumulator_in = tl.zeros((BLOCK_SIZE_B2, BLOCK_SIZE_N), dtype=tl.float32)
    y = tl.load(y_ptrs, mask=(offs_b1[:, None] < B1), other=0.0)
    s = tl.load(s_ptrs, mask=(offs_b1[None, :] < B1), other=0.0)
    accumulator_in = tl.dot(s, y, accumulator_in, allow_tf32=False)

    z = accumulator_in

    offs_zn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_zb2 = (tl.arange(0, BLOCK_SIZE_B2))
    z_ptrs = z_ptr + offs_zn[None, :] * stride_zn + offs_zb2[:, None] * stride_zb2 + pid_r * stride_zr
    z_mask = (offs_zn[None, :] < N) & (offs_zb2[:, None] < B2)
    tl.store(z_ptrs, z, mask=z_mask)

@triton.autotune(configs=_get_triton_blast_bmm_sxv_kernel_autotune_config(), key=['N', 'B2', 'R', 'B1'])
@triton.jit
def _triton_blast_bmm_sxv_kernel_fp16(
    y_ptr, s_ptr, z_ptr,
    N, B1, B2, R,
    stride_sr, stride_sb2, stride_sb1,
    stride_yr, stride_yb1, stride_yn,
    stride_zb2, stride_zr, stride_zn,
    BLOCK_SIZE_B2: tl.constexpr, BLOCK_SIZE_B1: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):

    """
    Triton kernel for batched matrix multiplication: Z[b2, r, n] = S[r, b2, b1] @ Y[r, b1, n]

    This kernel computes a batched matrix multiplication where each output tile Z[b2, r, n]
    is the result of multiplying a tensor S[r, b2, b1] with a tensor Y[r, b1, n].

    Parameters:
    -----------
    y_ptr : tl.pointer
        Pointer to the input tensor Y of shape (R, B1, N), stored in BF16.
    s_ptr : tl.pointer
        Pointer to the input tensor S of shape (R, B2, B1), stored in BF16.
    z_ptr : tl.pointer
        Pointer to the output tensor Z of shape (B2, R, N), stored in BF16.

    Notes:
    ------
    - Assumes B1, B2 >= 16, power of two, and not too large (~ <= 128)
    """

    pid_n = tl.program_id(axis=0)
    pid_r = tl.program_id(axis=1)

    offs_yn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_b2 = (tl.arange(0, BLOCK_SIZE_B2)) % B2
    offs_b1 = tl.arange(0, BLOCK_SIZE_B1)

    y_ptrs = y_ptr + (offs_yn[None, :] * stride_yn + offs_b1[:, None] * stride_yb1 + pid_r * stride_yr)
    s_ptrs = s_ptr + (offs_b2[:, None] * stride_sb2 + offs_b1[None, :] * stride_sb1 + pid_r * stride_sr)

    accumulator_in = tl.zeros((BLOCK_SIZE_B2, BLOCK_SIZE_N), dtype=tl.float32)
    y = tl.load(y_ptrs, mask=(offs_b1[:, None] < B1), other=0.0)
    s = tl.load(s_ptrs, mask=(offs_b1[None, :] < B1), other=0.0)
    accumulator_in = tl.dot(s, y, accumulator_in)

    z = accumulator_in.to(tl.bfloat16)

    offs_zn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_zb2 = (tl.arange(0, BLOCK_SIZE_B2))
    z_ptrs = z_ptr + offs_zn[None, :] * stride_zn + offs_zb2[:, None] * stride_zb2 + pid_r * stride_zr
    z_mask = (offs_zn[None, :] < N) & (offs_zb2[:, None] < B2)
    tl.store(z_ptrs, z, mask=z_mask)

@triton.jit
def _triton_blast_bmm_sxv_kernel_fp16_no_autotune(
    y_ptr, s_ptr, z_ptr,
    N, B1, B2, R,
    stride_sr, stride_sb2, stride_sb1,
    stride_yr, stride_yb1, stride_yn,
    stride_zb2, stride_zr, stride_zn,
    BLOCK_SIZE_B2: tl.constexpr, BLOCK_SIZE_B1: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):

    """
    Triton kernel for batched matrix multiplication: Z[b2, r, n] = S[r, b2, b1] @ Y[r, b1, n]

    This kernel computes a batched matrix multiplication where each output tile Z[b2, r, n]
    is the result of multiplying a tensor S[r, b2, b1] with a tensor Y[r, b1, n].

    Parameters:
    -----------
    y_ptr : tl.pointer
        Pointer to the input tensor Y of shape (R, B1, N), stored in BF16.
    s_ptr : tl.pointer
        Pointer to the input tensor S of shape (R, B2, B1), stored in BF16.
    z_ptr : tl.pointer
        Pointer to the output tensor Z of shape (B2, R, N), stored in BF16.

    Notes:
    ------
    - Assumes B1, B2 >= 16, power of two, and not too large (~ <= 128)
    """

    pid_n = tl.program_id(axis=0)
    pid_r = tl.program_id(axis=1)

    offs_yn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_b2 = (tl.arange(0, BLOCK_SIZE_B2)) % B2
    offs_b1 = tl.arange(0, BLOCK_SIZE_B1)

    y_ptrs = y_ptr + (offs_yn[None, :] * stride_yn + offs_b1[:, None] * stride_yb1 + pid_r * stride_yr)
    s_ptrs = s_ptr + (offs_b2[:, None] * stride_sb2 + offs_b1[None, :] * stride_sb1 + pid_r * stride_sr)

    accumulator_in = tl.zeros((BLOCK_SIZE_B2, BLOCK_SIZE_N), dtype=tl.float32)
    y = tl.load(y_ptrs, mask=(offs_b1[:, None] < B1), other=0.0)
    s = tl.load(s_ptrs, mask=(offs_b1[None, :] < B1), other=0.0)
    accumulator_in = tl.dot(s, y, accumulator_in)

    z = accumulator_in.to(tl.bfloat16)

    offs_zn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_zb2 = (tl.arange(0, BLOCK_SIZE_B2))
    z_ptrs = z_ptr + offs_zn[None, :] * stride_zn + offs_zb2[:, None] * stride_zb2 + pid_r * stride_zr
    z_mask = (offs_zn[None, :] < N) & (offs_zb2[:, None] < B2)
    tl.store(z_ptrs, z, mask=z_mask)

#-----------------------------------
@triton.jit
def _triton_blast_bmm_usxv_kernel_fp32(
    z_ptr, u_ptr, out_ptr,
    N, R, B2, Q,
    stride_zb2, stride_zr, stride_zn,
    stride_ub2, stride_uq, stride_ur,
    stride_outn, stride_outb2, stride_outq,
    BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_Q: tl.constexpr):

    """
    Triton kernel for batched matrix multiplication: OUT[b2, q, n] = U[b2, q, r] @ Z[b2, r, n]

    This kernel performs a batched matrix multiplication where a tensor
    U is multiplied with a tensor Z to produce an output tensor OUT.

    Parameters:
    -----------
    z_ptr : tl.pointer
        Pointer to tensor Z of shape (B2, R, N), stored in FP32.
    u_ptr : tl.pointer
        Pointer to tensor U of shape (B2, Q, R), stored in FP32.
    out_ptr : tl.pointer
        Pointer to output tensor OUT of shape (B2, Q, N), stored in FP32.
    """

    pid = tl.program_id(axis=0)
    num_pid_q = tl.cdiv(Q, BLOCK_SIZE_Q) 
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_Q * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_q = group_id * GROUP_SIZE_Q
    group_size_q = min(num_pid_q - first_pid_q, GROUP_SIZE_Q)
    pid_q = first_pid_q + ((pid % num_pid_in_group) % group_size_q)
    pid_n = (pid % num_pid_in_group) // group_size_q

    b2 = tl.program_id(axis=1)
    offs_uq = (pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) % Q
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_r = tl.arange(0, BLOCK_SIZE_R)

    u_ptrs = u_ptr + (offs_uq[:, None] * stride_uq + offs_r[None, :] * stride_ur + b2 * stride_ub2)
    z_ptrs = z_ptr + (offs_r[:, None] * stride_zr + offs_n[None, :] * stride_zn + b2 * stride_zb2)
    accumulator_in = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_N), dtype=tl.float32)

    for r in range(0, tl.cdiv(R, BLOCK_SIZE_R)):
        u_mask = (offs_r[None, :] < (R - r * BLOCK_SIZE_R))
        z_mask = (offs_r[:, None] < (R - r * BLOCK_SIZE_R))
        u = tl.load(u_ptrs, mask=u_mask, other=0.0)
        z = tl.load(z_ptrs, mask=z_mask, other=0.0)
        accumulator_in = tl.dot(u, z, accumulator_in, allow_tf32=False)
        u_ptrs += BLOCK_SIZE_R * stride_ur
        z_ptrs += BLOCK_SIZE_R * stride_zr
    
    out = accumulator_in.trans()

    offs_outn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_outq = (pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q))
    out_ptrs = out_ptr + offs_outn[:, None] * stride_outn + offs_outq[None, :] * stride_outq + b2 * stride_outb2
    out_mask = (offs_outn[:, None] < N) & (offs_outq[None, :] < Q)
    tl.store(out_ptrs, out, mask=out_mask)

@triton.autotune(configs=_get_triton_blast_bmm_usxv_kernel_autotune_config(), key=['N', 'R', 'Q', 'B2'])
@triton.jit
def _triton_blast_bmm_usxv_kernel_fp16(
    z_ptr, u_ptr, out_ptr,
    N, R, B2, Q,
    stride_zb2, stride_zr, stride_zn,
    stride_ub2, stride_uq, stride_ur,
    stride_outn, stride_outb2, stride_outq,
    BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_Q: tl.constexpr):

    """
    Triton kernel for batched matrix multiplication: OUT[b2, q, n] = U[b2, q, r] @ Z[b2, r, n]

    This kernel performs a batched matrix multiplication where a tensor
    U is multiplied with a tensor Z to produce an output tensor OUT.

    Parameters:
    -----------
    z_ptr : tl.pointer
        Pointer to tensor Z of shape (B2, R, N), stored in BF16.
    u_ptr : tl.pointer
        Pointer to tensor U of shape (B2, Q, R), stored in BF16.
    out_ptr : tl.pointer
        Pointer to output tensor OUT of shape (B2, Q, N), stored in BF16.
    """

    pid = tl.program_id(axis=0)
    num_pid_q = tl.cdiv(Q, BLOCK_SIZE_Q) 
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_Q * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_q = group_id * GROUP_SIZE_Q
    group_size_q = min(num_pid_q - first_pid_q, GROUP_SIZE_Q)
    pid_q = first_pid_q + ((pid % num_pid_in_group) % group_size_q)
    pid_n = (pid % num_pid_in_group) // group_size_q

    b2 = tl.program_id(axis=1)
    offs_uq = (pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) % Q
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_r = tl.arange(0, BLOCK_SIZE_R)

    u_ptrs = u_ptr + (offs_uq[:, None] * stride_uq + offs_r[None, :] * stride_ur + b2 * stride_ub2)
    z_ptrs = z_ptr + (offs_r[:, None] * stride_zr + offs_n[None, :] * stride_zn + b2 * stride_zb2)
    accumulator_in = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_N), dtype=tl.float32)

    for r in range(0, tl.cdiv(R, BLOCK_SIZE_R)):
        u_mask = (offs_r[None, :] < (R - r * BLOCK_SIZE_R))
        z_mask = (offs_r[:, None] < (R - r * BLOCK_SIZE_R))
        u = tl.load(u_ptrs, mask=u_mask, other=0.0)
        z = tl.load(z_ptrs, mask=z_mask, other=0.0)
        accumulator_in = tl.dot(u, z, accumulator_in)
        u_ptrs += BLOCK_SIZE_R * stride_ur
        z_ptrs += BLOCK_SIZE_R * stride_zr
    
    out = accumulator_in.to(tl.bfloat16).trans()

    offs_outn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_outq = (pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q))
    out_ptrs = out_ptr + offs_outn[:, None] * stride_outn + offs_outq[None, :] * stride_outq + b2 * stride_outb2
    out_mask = (offs_outn[:, None] < N) & (offs_outq[None, :] < Q)
    tl.store(out_ptrs, out, mask=out_mask)

@triton.jit
def _triton_blast_bmm_usxv_kernel_fp16_no_autotune(
    z_ptr, u_ptr, out_ptr,
    N, R, B2, Q,
    stride_zb2, stride_zr, stride_zn,
    stride_ub2, stride_uq, stride_ur,
    stride_outn, stride_outb2, stride_outq,
    BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_Q: tl.constexpr):

    """
    Triton kernel for batched matrix multiplication: OUT[b2, q, n] = U[b2, q, r] @ Z[b2, r, n]

    This kernel performs a batched matrix multiplication where a tensor
    U is multiplied with a tensor Z to produce an output tensor OUT.

    Parameters:
    -----------
    z_ptr : tl.pointer
        Pointer to tensor Z of shape (B2, R, N), stored in BF16.
    u_ptr : tl.pointer
        Pointer to tensor U of shape (B2, Q, R), stored in BF16.
    out_ptr : tl.pointer
        Pointer to output tensor OUT of shape (B2, Q, N), stored in BF16.
    """

    pid = tl.program_id(axis=0)
    num_pid_q = tl.cdiv(Q, BLOCK_SIZE_Q) 
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_Q * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_q = group_id * GROUP_SIZE_Q
    group_size_q = min(num_pid_q - first_pid_q, GROUP_SIZE_Q)
    pid_q = first_pid_q + ((pid % num_pid_in_group) % group_size_q)
    pid_n = (pid % num_pid_in_group) // group_size_q

    b2 = tl.program_id(axis=1)
    offs_uq = (pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) % Q
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_r = tl.arange(0, BLOCK_SIZE_R)

    u_ptrs = u_ptr + (offs_uq[:, None] * stride_uq + offs_r[None, :] * stride_ur + b2 * stride_ub2)
    z_ptrs = z_ptr + (offs_r[:, None] * stride_zr + offs_n[None, :] * stride_zn + b2 * stride_zb2)
    accumulator_in = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_N), dtype=tl.float32)

    for r in range(0, tl.cdiv(R, BLOCK_SIZE_R)):
        u_mask = (offs_r[None, :] < (R - r * BLOCK_SIZE_R))
        z_mask = (offs_r[:, None] < (R - r * BLOCK_SIZE_R))
        u = tl.load(u_ptrs, mask=u_mask, other=0.0)
        z = tl.load(z_ptrs, mask=z_mask, other=0.0)
        accumulator_in = tl.dot(u, z, accumulator_in)
        u_ptrs += BLOCK_SIZE_R * stride_ur
        z_ptrs += BLOCK_SIZE_R * stride_zr
    
    out = accumulator_in.to(tl.bfloat16).trans()

    offs_outn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_outq = (pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q))
    out_ptrs = out_ptr + offs_outn[:, None] * stride_outn + offs_outq[None, :] * stride_outq + b2 * stride_outb2
    out_mask = (offs_outn[:, None] < N) & (offs_outq[None, :] < Q)
    tl.store(out_ptrs, out, mask=out_mask)

""" Triton BLAST Kernel Launchers """
#-----------------------------------
def _triton_blast_partial_launcher_fp32(
    x: torch.Tensor, 
    v: torch.Tensor, 
    s: torch.Tensor,
    best_config: triton.Config) -> Tuple[torch.Tensor, ...]:

    """
    Launches the triton_blast_partial_kernel_fp32 Triton kernel

    Parameters:
    ----------
    x : torch.Tensor
        Input matrix of shape (B1, N, P)
    v : torch.Tensor
        Weight matrix of shape (B1, P, R), must be contiguous
    s : torch.Tensor
        Scaling matrix of shape (B1, B2, R), must be contiguous

    Returns:
    -------
    z : torch.Tensor
        The accumulated output matrix of shape (B2, N, R)
    y : torch.Tensor
        The intermediate result matrix of shape (B1, N, R)
    """

    assert x.shape[2] == v.shape[1] and x.shape[0] == v.shape[0], "Incompatible dimensions X and V"
    assert x.shape[0] == s.shape[0] and v.shape[2] == s.shape[2], "Incompatible dimensions X and S"
    assert v.is_contiguous(), "Matrix V must be contiguous"
    assert s.is_contiguous(), "Matrix S must be contiguous"
    assert x.dtype == torch.float32
    assert v.dtype == torch.float32
    assert s.dtype == torch.float32

    B1, N, P = x.shape
    B1, P, R = v.shape
    B1, B2, R = s.shape
    y = torch.empty((B1, N, R), device=x.device, dtype=x.dtype)
    z = torch.empty((B2, N, R), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(N, best_config.kwargs['BLOCK_SIZE_N']), )
    _triton_blast_partial_kernel_fp32[grid](
        x, v, s, z, y,
        N, P, B1, R, B2,
        x.stride(0), x.stride(1), x.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        s.stride(0), s.stride(1), s.stride(2),
        z.stride(0), z.stride(1), z.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_SIZE_N=best_config.kwargs['BLOCK_SIZE_N'],
        BLOCK_SIZE_P=best_config.kwargs['BLOCK_SIZE_P'],
        BLOCK_SIZE_B1=next_power_of_2(B1), 
        BLOCK_SIZE_R=next_power_of_2(R), 
        BLOCK_SIZE_B2=next_power_of_2(B2),
        num_stages=1,
        num_warps=best_config.num_warps
    )
    return z, y

def _triton_blast_partial_launcher_fp16(
    x: torch.Tensor, 
    v: torch.Tensor, 
    s: torch.Tensor,
    config: triton.Config = None) -> torch.Tensor:
    
    """
    Launches the triton_blast_partial_kernel_fp16 Triton kernel

    Parameters:
    ----------
    x : torch.Tensor
        Input matrix of shape (B1, N, P)
    v : torch.Tensor
        Weight matrix of shape (B1, P, R), must be contiguous
    s : torch.Tensor
        Scaling matrix of shape (B1, B2, R), must be contiguous

    Returns:
    -------
    z : torch.Tensor
        The accumulated output matrix of shape (B2, N, R)
    """
    
    assert x.shape[2] == v.shape[1] and x.shape[0] == v.shape[0], "Incompatible dimensions X and V"
    assert x.shape[0] == s.shape[0] and v.shape[2] == s.shape[2], "Incompatible dimensions X and S"
    assert v.is_contiguous(), "Matrix V must be contiguous"
    assert s.is_contiguous(), "Matrix S must be contiguous"
    assert x.dtype == torch.bfloat16
    assert v.dtype == torch.bfloat16
    assert s.dtype == torch.bfloat16

    B1, N, P = x.shape
    B1, P, R = v.shape
    B1, B2, R = s.shape
    z = torch.empty((B2, N, R), device=x.device, dtype=x.dtype)
    if config is None:
        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']), )
        _triton_blast_partial_kernel_fp16[grid](
            x, v, s, z,
            N, P, B1, R, B2,
            x.stride(0), x.stride(1), x.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            s.stride(0), s.stride(1), s.stride(2),
            z.stride(0), z.stride(1), z.stride(2),
            BLOCK_SIZE_B1=next_power_of_2(B1), BLOCK_SIZE_R=next_power_of_2(R), BLOCK_SIZE_B2=next_power_of_2(B2)
        )
    else:
        grid = (triton.cdiv(N, config.kwargs['BLOCK_SIZE_N']), )
        _triton_blast_partial_kernel_fp16_no_autotune[grid](
            x, v, s, z,
            N, P, B1, R, B2,
            x.stride(0), x.stride(1), x.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            s.stride(0), s.stride(1), s.stride(2),
            z.stride(0), z.stride(1), z.stride(2),
            BLOCK_SIZE_N=config.kwargs['BLOCK_SIZE_N'],
            BLOCK_SIZE_P=config.kwargs['BLOCK_SIZE_P'],
            BLOCK_SIZE_B1=next_power_of_2(B1), 
            BLOCK_SIZE_R=next_power_of_2(R), 
            BLOCK_SIZE_B2=next_power_of_2(B2),
            num_stages=config.num_stages,
            num_warps=config.num_warps
        )
    return z

#-----------------------------------
def _triton_blast_partial_grouped_launcher_fp32(
    x: torch.Tensor, 
    v: torch.Tensor, 
    s: torch.Tensor,
    best_config: triton.Config) -> Tuple[torch.Tensor, ...]:
    
    """
    Launches the triton_blast_partial_grouped_kernel_fp32 Triton kernel

    Parameters:
    ----------
    x : torch.Tensor
        Input matrix of shape (B1, N, P)
    v : torch.Tensor
        Weight matrix of shape (B1, P, R), must be contiguous
    s : torch.Tensor
        Scaling matrix of shape (B1, B2, R), must be contiguous

    Returns:
    -------
    z : torch.Tensor
        The accumulated output matrix of shape (B2, N, R)
    y : torch.Tensor
        The intermediate result matrix of shape (B1, N, R)
    """
    
    assert x.shape[2] == v.shape[1] and x.shape[0] == v.shape[0], "Incompatible dimensions X and V"
    assert x.shape[0] == s.shape[0] and v.shape[2] == s.shape[2], "Incompatible dimensions X and S"
    assert v.is_contiguous(), "Matrix V must be contiguous"
    assert s.is_contiguous(), "Matrix S must be contiguous"
    assert x.dtype == torch.float32
    assert v.dtype == torch.float32
    assert s.dtype == torch.float32

    B1, N, P = x.shape
    B1, P, R = v.shape
    B1, B2, R = s.shape
    y = torch.empty((B1, N, R), device=x.device, dtype=x.dtype)
    z = torch.empty((B2, N, R), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(N, best_config.kwargs['BLOCK_SIZE_N']) * triton.cdiv(R, best_config.kwargs['BLOCK_SIZE_R']), )
    _triton_blast_partial_grouped_kernel_fp32[grid](
        x, v, s, z, y,
        N, P, B1, R, B2,
        x.stride(0), x.stride(1), x.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        s.stride(0), s.stride(1), s.stride(2),
        z.stride(0), z.stride(1), z.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_SIZE_N=best_config.kwargs['BLOCK_SIZE_N'],
        BLOCK_SIZE_R=best_config.kwargs['BLOCK_SIZE_R'],
        BLOCK_SIZE_P=best_config.kwargs['BLOCK_SIZE_P'],
        GROUP_SIZE_N=best_config.kwargs['GROUP_SIZE_N'],
        BLOCK_SIZE_B1=next_power_of_2(B1), 
        BLOCK_SIZE_B2=next_power_of_2(B2), 
        num_stages=best_config.num_stages,
        num_warps=best_config.num_warps
    )
    return z, y

def _triton_blast_partial_grouped_launcher_fp16(
    x: torch.Tensor, 
    v: torch.Tensor, 
    s: torch.Tensor, # ) -> torch.Tensor:
    config: triton.Config = None) -> torch.Tensor:
    
    """
    Launches the triton_blast_partial_kernel_fp16 Triton kernel

    Parameters:
    ----------
    x : torch.Tensor
        Input matrix of shape (B1, N, P)
    v : torch.Tensor
        Weight matrix of shape (B1, P, R), must be contiguous
    s : torch.Tensor
        Scaling matrix of shape (B1, B2, R), must be contiguous

    Returns:
    -------
    z : torch.Tensor
        The accumulated output matrix of shape (B2, N, R)
    """
    
    assert x.shape[2] == v.shape[1] and x.shape[0] == v.shape[0], "Incompatible dimensions X and V"
    assert x.shape[0] == s.shape[0] and v.shape[2] == s.shape[2], "Incompatible dimensions X and S"
    assert v.is_contiguous(), "Matrix V must be contiguous"
    assert s.is_contiguous(), "Matrix S must be contiguous"
    assert x.dtype == torch.bfloat16
    assert v.dtype == torch.bfloat16
    assert s.dtype == torch.bfloat16

    B1, N, P = x.shape
    B1, P, R = v.shape
    B1, B2, R = s.shape
    z = torch.empty((B2, N, R), device=x.device, dtype=x.dtype)
    if config is None:
        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']) * triton.cdiv(R, META['BLOCK_SIZE_R']), )
        _triton_blast_partial_grouped_kernel_fp16[grid](
            x, v, s, z,
            N, P, B1, R, B2,
            x.stride(0), x.stride(1), x.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            s.stride(0), s.stride(1), s.stride(2),
            z.stride(0), z.stride(1), z.stride(2),
            BLOCK_SIZE_B1=next_power_of_2(B1), BLOCK_SIZE_B2=next_power_of_2(B2)
        )
    else:
        grid = (triton.cdiv(N, config.kwargs['BLOCK_SIZE_N']) * triton.cdiv(R, config.kwargs['BLOCK_SIZE_R']), )
        _triton_blast_partial_grouped_kernel_fp16_no_autotune[grid](
            x, v, s, z,
            N, P, B1, R, B2,
            x.stride(0), x.stride(1), x.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            s.stride(0), s.stride(1), s.stride(2),
            z.stride(0), z.stride(1), z.stride(2),
            BLOCK_SIZE_N=config.kwargs['BLOCK_SIZE_N'],
            BLOCK_SIZE_R=config.kwargs['BLOCK_SIZE_R'],
            BLOCK_SIZE_P=config.kwargs['BLOCK_SIZE_P'],
            GROUP_SIZE_N=config.kwargs['GROUP_SIZE_N'],
            BLOCK_SIZE_B1=next_power_of_2(B1), 
            BLOCK_SIZE_B2=next_power_of_2(B2), 
            num_stages=config.num_stages,
            num_warps=config.num_warps
        )
    return z

#-----------------------------------
def _triton_blast_partial_grouped_persistent_launcher_fp32(
    x: torch.Tensor, 
    v: torch.Tensor, 
    s: torch.Tensor,
    best_config: triton.Config) -> Tuple[torch.Tensor, ...]:
                        
    """
    Launches the triton_blast_partial_grouped_persistent_kernel_fp32 Triton kernel

    Parameters:
    ----------
    x : torch.Tensor
        Input matrix of shape (B1, N, P)
    v : torch.Tensor
        Weight matrix of shape (B1, P, R), must be contiguous
    s : torch.Tensor
        Scaling matrix of shape (B1, B2, R), must be contiguous

    Returns:
    -------
    z : torch.Tensor
        The accumulated output matrix of shape (B2, N, R)
    y : torch.Tensor
        The intermediate result matrix of shape (B1, N, R)
    """

    assert x.shape[2] == v.shape[1] and x.shape[0] == v.shape[0], "Incompatible dimensions X and V"
    assert x.shape[0] == s.shape[0] and v.shape[2] == s.shape[2], "Incompatible dimensions X and S"
    assert v.is_contiguous(), "Matrix V must be contiguous"
    assert s.is_contiguous(), "Matrix S must be contiguous"
    assert x.dtype == torch.float32
    assert v.dtype == torch.float32
    assert s.dtype == torch.float32

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    B1, N, P = x.shape
    B1, P, R = v.shape
    B1, B2, R = s.shape
    y = torch.empty((B1, N, R), device=x.device, dtype=x.dtype)
    z = torch.empty((B2, N, R), device=x.device, dtype=x.dtype)

    grid = (min(NUM_SMS, triton.cdiv(N, best_config.kwargs['BLOCK_SIZE_N']) * triton.cdiv(R, best_config.kwargs['BLOCK_SIZE_R'])), )
    _triton_blast_partial_grouped_persistent_kernel_fp32[grid](
        x, v, s, z, y,
        N, P, B1, R, B2,
        x.stride(0), x.stride(1), x.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        s.stride(0), s.stride(1), s.stride(2),
        z.stride(0), z.stride(1), z.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_SIZE_N=best_config.kwargs['BLOCK_SIZE_N'],
        BLOCK_SIZE_R=best_config.kwargs['BLOCK_SIZE_R'],
        BLOCK_SIZE_P=best_config.kwargs['BLOCK_SIZE_P'],
        GROUP_SIZE_N=best_config.kwargs['GROUP_SIZE_N'],
        BLOCK_SIZE_B1=next_power_of_2(B1), 
        BLOCK_SIZE_B2=next_power_of_2(B2), 
        NUM_SMS=NUM_SMS,
        num_stages=best_config.num_stages,
        num_warps=best_config.num_warps
    )
    return z, y

def _triton_blast_partial_grouped_persistent_launcher_fp16(
    x: torch.Tensor, 
    v: torch.Tensor, 
    s: torch.Tensor,
    config: triton.Config = None) -> torch.Tensor:
    
    """
    Launches the triton_blast_partial_grouped_persistent_kernel_fp16 Triton kernel

    Parameters:
    ----------
    x : torch.Tensor
        Input matrix of shape (B1, N, P)
    v : torch.Tensor
        Weight matrix of shape (B1, P, R), must be contiguous
    s : torch.Tensor
        Scaling matrix of shape (B1, B2, R), must be contiguous

    Returns:
    -------
    z : torch.Tensor
        The accumulated output matrix of shape (B2, N, R)
    """
    
    assert x.shape[2] == v.shape[1] and x.shape[0] == v.shape[0], "Incompatible dimensions X and V"
    assert x.shape[0] == s.shape[0] and v.shape[2] == s.shape[2], "Incompatible dimensions X and S"
    assert v.is_contiguous(), "Matrix V must be contiguous"
    assert s.is_contiguous(), "Matrix S must be contiguous"
    assert x.dtype == torch.bfloat16
    assert v.dtype == torch.bfloat16
    assert s.dtype == torch.bfloat16

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    B1, N, P = x.shape
    B1, P, R = v.shape
    B1, B2, R = s.shape
    z = torch.empty((B2, N, R), device=x.device, dtype=x.dtype)
    if config is None:
        grid = lambda META: (min(NUM_SMS, triton.cdiv(N, META['BLOCK_SIZE_N']) * triton.cdiv(R, META['BLOCK_SIZE_R'])), )
        _triton_blast_partial_grouped_persistent_kernel_fp16[grid](
            x, v, s, z,
            N, P, B1, R, B2,
            x.stride(0), x.stride(1), x.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            s.stride(0), s.stride(1), s.stride(2),
            z.stride(0), z.stride(1), z.stride(2),
            BLOCK_SIZE_B1=next_power_of_2(B1), BLOCK_SIZE_B2=next_power_of_2(B2), NUM_SMS=NUM_SMS
        )
    else:
       grid = (min(NUM_SMS, triton.cdiv(N, config.kwargs['BLOCK_SIZE_N']) * triton.cdiv(R, config.kwargs['BLOCK_SIZE_R'])), )
       _triton_blast_partial_grouped_persistent_kernel_fp16_no_autotune[grid](
            x, v, s, z,
            N, P, B1, R, B2,
            x.stride(0), x.stride(1), x.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            s.stride(0), s.stride(1), s.stride(2),
            z.stride(0), z.stride(1), z.stride(2),
            BLOCK_SIZE_N=config.kwargs['BLOCK_SIZE_N'],
            BLOCK_SIZE_R=config.kwargs['BLOCK_SIZE_R'],
            BLOCK_SIZE_P=config.kwargs['BLOCK_SIZE_P'],
            GROUP_SIZE_N=config.kwargs['GROUP_SIZE_N'],
            BLOCK_SIZE_B1=next_power_of_2(B1), 
            BLOCK_SIZE_B2=next_power_of_2(B2), 
            NUM_SMS=NUM_SMS,
            num_stages=config.num_stages,
            num_warps=config.num_warps
        )
    return z

#-----------------------------------
def _triton_blast_full_launcher_fp32(
    x: torch.Tensor, 
    v: torch.Tensor, 
    s: torch.Tensor,
    u: torch.Tensor,
    best_config: triton.Config) -> Tuple[torch.Tensor, ...]:
    
    """
    Launches the triton_blast_full_fp32 Triton kernel

    Parameters:
    ----------
    x : torch.Tensor
        Input matrix of shape (B1, N, P)
    v : torch.Tensor
        Weight matrix of shape (B1, P, R), must be contiguous
    s : torch.Tensor
        Scaling matrix of shape (B1, B2, R), must be contiguous
    u : torch.Tensor
        Transformation matrix of shape (B2, R, Q), must be contiguous

    Returns:
    -------
    o : torch.Tensor
        The final output matrix of shape (B2, N, Q)
    z : torch.Tensor
        The intermediate accumulated matrix of shape (B2, N, R)
    y : torch.Tensor
        The intermediate matrix multiplication result of shape (B1, N, R)
    """

    assert x.shape[2] == v.shape[1] and x.shape[0] == v.shape[0], "Incompatible X and V dimensions"
    assert x.shape[0] == s.shape[0] and v.shape[2] == s.shape[2], "Incompatible S dimensions"
    assert s.shape[2] == u.shape[1] and s.shape[1] == u.shape[0], "Incompatible U dimensions" 
    assert v.is_contiguous(), "Matrix V must be contiguous"
    assert s.is_contiguous(), "Matrix S must be contiguous"
    assert u.is_contiguous(), "Matrix U must be contiguous"
    assert x.dtype == torch.float32
    assert v.dtype == torch.float32
    assert s.dtype == torch.float32
    assert u.dtype == torch.float32

    B1, N, P = x.shape
    B1, P, R = v.shape
    B1, B2, R = s.shape
    B2, R, Q = u.shape
    y = torch.empty((B1, N, R), device=x.device, dtype=x.dtype)
    z = torch.empty((B2, N, R), device=x.device, dtype=x.dtype)
    o = torch.empty((B2, N, Q), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(N, best_config.kwargs['BLOCK_SIZE_N']), )
    _triton_blast_full_kernel_fp32[grid](
        x, v, s, u, o, z, y,
        N, P, B1, R, B2, Q,
        x.stride(0), x.stride(1), x.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        s.stride(0), s.stride(1), s.stride(2),
        u.stride(0), u.stride(1), u.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        z.stride(0), z.stride(1), z.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_SIZE_N=best_config.kwargs['BLOCK_SIZE_N'],
        BLOCK_SIZE_Q=best_config.kwargs['BLOCK_SIZE_Q'],
        BLOCK_SIZE_P=best_config.kwargs['BLOCK_SIZE_P'],
        BLOCK_SIZE_B1=next_power_of_2(B1), 
        BLOCK_SIZE_R=next_power_of_2(R), 
        BLOCK_SIZE_B2=next_power_of_2(B2),
        num_warps=best_config.num_warps,
        num_stages=best_config.num_stages
    )
    return o, z, y

def _triton_blast_full_launcher_fp16(
    x: torch.Tensor, 
    v: torch.Tensor, 
    s: torch.Tensor,
    u: torch.Tensor,
    config: triton.Config = None) -> torch.Tensor:

    """
    Launches the triton_blast_full_fp16 Triton kernel

    Parameters:
    ----------
    x : torch.Tensor
        Input matrix of shape (B1, N, P)
    v : torch.Tensor
        Weight matrix of shape (B1, P, R), must be contiguous
    s : torch.Tensor
        Scaling matrix of shape (B1, B2, R), must be contiguous
    u : torch.Tensor
        Transformation matrix of shape (B2, R, Q), must be contiguous

    Returns:
    -------
    o : torch.Tensor
        The final output matrix of shape (B2, N, Q)
    """

    assert x.shape[2] == v.shape[1] and x.shape[0] == v.shape[0], "Incompatible X and V dimensions"
    assert x.shape[0] == s.shape[0] and v.shape[2] == s.shape[2], "Incompatible S dimensions"
    assert s.shape[2] == u.shape[1] and s.shape[1] == u.shape[0], "Incompatible U dimensions" 
    assert v.is_contiguous(), "Matrix V must be contiguous"
    assert s.is_contiguous(), "Matrix S must be contiguous"
    assert u.is_contiguous(), "Matrix U must be contiguous"
    assert x.dtype == torch.bfloat16
    assert v.dtype == torch.bfloat16
    assert s.dtype == torch.bfloat16
    assert u.dtype == torch.bfloat16

    B1, N, P = x.shape
    B1, P, R = v.shape
    B1, B2, R = s.shape
    B2, R, Q = u.shape
    o = torch.empty((B2, N, Q), device=x.device, dtype=x.dtype)
    if config is None:
        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']), )
        _triton_blast_full_kernel_fp16[grid](
            x, v, s, u, o,
            N, P, B1, R, B2, Q,
            x.stride(0), x.stride(1), x.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            s.stride(0), s.stride(1), s.stride(2),
            u.stride(0), u.stride(1), u.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            BLOCK_SIZE_B1=next_power_of_2(B1), BLOCK_SIZE_R=next_power_of_2(R), BLOCK_SIZE_B2=next_power_of_2(B2)
        )
    else:
        grid = (triton.cdiv(N, config.kwargs['BLOCK_SIZE_N']), )
        _triton_blast_full_kernel_fp16_no_autotune[grid](
            x, v, s, u, o,
            N, P, B1, R, B2, Q,
            x.stride(0), x.stride(1), x.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            s.stride(0), s.stride(1), s.stride(2),
            u.stride(0), u.stride(1), u.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            BLOCK_SIZE_N=config.kwargs['BLOCK_SIZE_N'],
            BLOCK_SIZE_Q=config.kwargs['BLOCK_SIZE_Q'],
            BLOCK_SIZE_P=config.kwargs['BLOCK_SIZE_P'],
            BLOCK_SIZE_B1=next_power_of_2(B1), 
            BLOCK_SIZE_R=next_power_of_2(R), 
            BLOCK_SIZE_B2=next_power_of_2(B2),
            num_warps=config.num_warps,
            num_stages=config.num_stages
        )
    return o

#-----------------------------------
def _triton_blast_bmm_xv_launcher_fp32(
    x: torch.Tensor, 
    v: torch.Tensor,
    best_config: triton.Config) -> torch.Tensor:
    
    assert x.shape[2] == v.shape[1] and x.shape[0] == v.shape[0], "Incompatible dimensions X and V"
    assert v.is_contiguous(), "Matrix V must be contiguous"
    assert x.dtype == torch.float32
    assert v.dtype == torch.float32

    B1, N, P = x.shape
    B1, P, R = v.shape

    y = torch.empty((R, B1, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(N, best_config.kwargs['BLOCK_SIZE_N']) * triton.cdiv(R, best_config.kwargs['BLOCK_SIZE_R']), B1)
    _triton_blast_bmm_xv_kernel_fp32[grid](
        x, v, y,
        N, P, B1, R,
        x.stride(0), x.stride(1), x.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_SIZE_N=best_config.kwargs['BLOCK_SIZE_N'], 
        BLOCK_SIZE_P=best_config.kwargs['BLOCK_SIZE_P'], 
        BLOCK_SIZE_R=best_config.kwargs['BLOCK_SIZE_R'],
        GROUP_SIZE_N=best_config.kwargs['GROUP_SIZE_N']
    )

    return y

def _triton_blast_bmm_xv_launcher_fp16(
    x: torch.Tensor, 
    v: torch.Tensor,
    config: triton.Config = None) -> torch.Tensor:
    
    assert x.shape[2] == v.shape[1] and x.shape[0] == v.shape[0], "Incompatible dimensions X and V"
    assert v.is_contiguous(), "Matrix V must be contiguous"
    assert x.dtype == torch.bfloat16
    assert v.dtype == torch.bfloat16

    B1, N, P = x.shape
    B1, P, R = v.shape

    y = torch.empty((R, B1, N), device=x.device, dtype=x.dtype)

    if config is None:
        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']) * triton.cdiv(R, META['BLOCK_SIZE_R']), B1)
        _triton_blast_bmm_xv_kernel_fp16[grid](
            x, v, y,
            N, P, B1, R,
            x.stride(0), x.stride(1), x.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            y.stride(0), y.stride(1), y.stride(2)
        )
    else:
        grid = (triton.cdiv(N, config.kwargs['BLOCK_SIZE_N']) * triton.cdiv(R, config.kwargs['BLOCK_SIZE_R']), B1)
        _triton_blast_bmm_xv_kernel_fp16_no_autotune[grid](
            x, v, y,
            N, P, B1, R,
            x.stride(0), x.stride(1), x.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_SIZE_N=config.kwargs['BLOCK_SIZE_N'], 
            BLOCK_SIZE_P=config.kwargs['BLOCK_SIZE_P'], 
            BLOCK_SIZE_R=config.kwargs['BLOCK_SIZE_R'],
            GROUP_SIZE_N=config.kwargs['GROUP_SIZE_N']
        )
    return y

#-----------------------------------
def _triton_blast_bmm_sxv_launcher_fp32(
    y: torch.Tensor, 
    s: torch.Tensor,
    best_config: triton.Config) -> torch.Tensor:
    
    assert y.shape[1] == s.shape[2] and y.shape[0] == s.shape[0], "Incompatible dimensions Y and S"
    assert s.is_contiguous(), "Matrix S must be contiguous"
    assert y.dtype == torch.float32
    assert s.dtype == torch.float32

    R, B1, N = y.shape
    R, B2, B1 = s.shape

    z = torch.empty((B2, R, N), device=y.device, dtype=y.dtype)

    BLOCK_SIZE_B1 = max(next_power_of_2(B1), 16)
    BLOCK_SIZE_B2 = max(next_power_of_2(B2), 16)
    
    grid = (triton.cdiv(N, best_config.kwargs['BLOCK_SIZE_N']), R)
    
    _triton_blast_bmm_sxv_kernel_fp32[grid](
        y, s, z,
        N, B1, B2, R,
        s.stride(0), s.stride(1), s.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        z.stride(0), z.stride(1), z.stride(2),
        BLOCK_SIZE_B2=BLOCK_SIZE_B2,
        BLOCK_SIZE_B1=BLOCK_SIZE_B1,
        BLOCK_SIZE_N=best_config.kwargs['BLOCK_SIZE_N']
    )
    return z

def _triton_blast_bmm_sxv_launcher_fp16(
    y: torch.Tensor, 
    s: torch.Tensor,
    config: triton.Config = None) -> torch.Tensor:
    
    assert y.shape[1] == s.shape[2] and y.shape[0] == s.shape[0], "Incompatible dimensions Y and S"
    assert s.is_contiguous(), "Matrix S must be contiguous"
    assert y.dtype == torch.bfloat16
    assert s.dtype == torch.bfloat16

    R, B1, N = y.shape
    R, B2, B1 = s.shape

    z = torch.empty((B2, R, N), device=y.device, dtype=y.dtype)

    BLOCK_SIZE_B1 = max(next_power_of_2(B1), 16)
    BLOCK_SIZE_B2 = max(next_power_of_2(B2), 16)

    if config is None:
        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']), R)
        _triton_blast_bmm_sxv_kernel_fp16[grid](
            y, s, z,
            N, B1, B2, R,
            s.stride(0), s.stride(1), s.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            z.stride(0), z.stride(1), z.stride(2),
            BLOCK_SIZE_B2=BLOCK_SIZE_B2,
            BLOCK_SIZE_B1=BLOCK_SIZE_B1
        )
    else:
        grid = (triton.cdiv(N, config.kwargs['BLOCK_SIZE_N']), R)
        _triton_blast_bmm_sxv_kernel_fp16_no_autotune[grid](
            y, s, z,
            N, B1, B2, R,
            s.stride(0), s.stride(1), s.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            z.stride(0), z.stride(1), z.stride(2),
            BLOCK_SIZE_B2=BLOCK_SIZE_B2,
            BLOCK_SIZE_B1=BLOCK_SIZE_B1,
            BLOCK_SIZE_N=config.kwargs['BLOCK_SIZE_N']
        )
    return z

#-----------------------------------
def _triton_blast_bmm_usxv_launcher_fp32(
    z: torch.Tensor,
    u: torch.Tensor,
    best_config: triton.Config) -> torch.Tensor: 
    
    assert u.shape[2] == z.shape[1] and z.shape[0] == u.shape[0], "Incompatible dimensions Z and U"
    assert u.is_contiguous(), "Matrix U must be contiguous"
    assert z.dtype == torch.float32
    assert u.dtype == torch.float32

    B2, R, N = z.shape
    B2, Q, R = u.shape

    out = torch.empty((N, B2, Q), device=z.device, dtype=z.dtype)

    grid = (triton.cdiv(N, best_config.kwargs['BLOCK_SIZE_N']) * triton.cdiv(Q, best_config.kwargs['BLOCK_SIZE_Q']), B2)
    _triton_blast_bmm_usxv_kernel_fp32[grid](
        z, u, out,
        N, R, B2, Q,
        z.stride(0), z.stride(1), z.stride(2),
        u.stride(0), u.stride(1), u.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_SIZE_Q=best_config.kwargs['BLOCK_SIZE_Q'], 
        BLOCK_SIZE_R=best_config.kwargs['BLOCK_SIZE_R'],
        BLOCK_SIZE_N=best_config.kwargs['BLOCK_SIZE_N'],
        GROUP_SIZE_Q=best_config.kwargs['GROUP_SIZE_Q']
    )
    return out

def _triton_blast_bmm_usxv_launcher_fp16(
    z: torch.Tensor,
    u: torch.Tensor,
    config: triton.Config = None) -> torch.Tensor: 
    
    assert u.shape[2] == z.shape[1] and z.shape[0] == u.shape[0], "Incompatible dimensions Z and U"
    assert u.is_contiguous(), "Matrix U must be contiguous"
    assert z.dtype == torch.bfloat16
    assert u.dtype == torch.bfloat16

    B2, R, N = z.shape
    B2, Q, R = u.shape

    out = torch.empty((N, B2, Q), device=z.device, dtype=z.dtype)

    if config is None:
        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']) * triton.cdiv(Q, META['BLOCK_SIZE_Q']), B2)
        _triton_blast_bmm_usxv_kernel_fp16[grid](
            z, u, out,
            N, R, B2, Q,
            z.stride(0), z.stride(1), z.stride(2),
            u.stride(0), u.stride(1), u.stride(2),
            out.stride(0), out.stride(1), out.stride(2)
        )
    else:
        grid = (triton.cdiv(N, config.kwargs['BLOCK_SIZE_N']) * triton.cdiv(Q, config.kwargs['BLOCK_SIZE_Q']), B2)
        _triton_blast_bmm_usxv_kernel_fp16_no_autotune[grid](
            z, u, out,
            N, R, B2, Q,
            z.stride(0), z.stride(1), z.stride(2),
            u.stride(0), u.stride(1), u.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_SIZE_Q=config.kwargs['BLOCK_SIZE_Q'], 
            BLOCK_SIZE_R=config.kwargs['BLOCK_SIZE_R'],
            BLOCK_SIZE_N=config.kwargs['BLOCK_SIZE_N'],
            GROUP_SIZE_Q=config.kwargs['GROUP_SIZE_Q']
        )
    return out

""" Triton BLAST Functions """
#-----------------------------------
def triton_blast_partial_fp32(
    x: torch.Tensor, 
    U: torch.Tensor, 
    V: torch.Tensor, 
    S: torch.Tensor) -> Tuple[torch.Tensor, ...]:

    """
    Performs BLAST matrix multiplication using triton_blast_partial_launcher_fp32 and is
    useful for checking correctness of entire BLAST operation

    Parameters:
    ----------
    x : torch.Tensor
        Input tensor of shape (num_batches, num_seq, in_f), where in_f is the last dimension
    U : torch.Tensor
        Tensor of shape (b2, rank, out_f // b2)
    V : torch.Tensor
        Tensor of shape (b1, in_f // b1, rank)
    S : torch.Tensor
        Scaling tensor of shape (b1, b2, rank)

    Returns:
    -------
    out: torch.Tensor
        Output tensor of shape (num_batches, num_seq, out_f), resulting from the BLAST matrix multiplication
    """

    best_config = getattr(_triton_blast_partial_kernel_fp16, 'best_config', None)

    if best_config is None:
        default_config = triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_P': 32 }, num_stages=3, num_warps=4)
        print(f"No autotuned config found. Running with default config: {default_config}")
        print("Ensure to run triton_blast_partial_fp16 before")
        best_config = default_config

    b1, p, r = V.shape
    b2, _, q = U.shape
    assert b1 == b2
    assert b1 != 1 and b2 != 1

    x_shape = x.shape
    x = x.flatten(0, -2)
    x = x.view(-1, b1, x_shape[-1] // b1).transpose(0, 1)

    z, y = _triton_blast_partial_launcher_fp32(x, V, S, best_config)

    out = torch.empty(x_shape[0] * x_shape[1], b2, q, device=x.device, dtype=x.dtype).transpose(0, 1)
    out = torch.bmm(z, U, out=out)
    out = out.transpose(0, 1).reshape((x_shape[0], x_shape[1], b2 * q))

    return out, z, y

def triton_blast_partial_fp16(
    x: torch.Tensor, 
    U: torch.Tensor, 
    V: torch.Tensor, 
    S: torch.Tensor,
    config: triton.Config = None) -> torch.Tensor:

    """
    Performs BLAST matrix multiplication using triton_blast_partial_launcher_fp16 and is
    useful for checking correctness of entire BLAST operation

    Parameters:
    ----------
    x : torch.Tensor
        Input tensor of shape (num_batches, num_seq, in_f), where in_f is the last dimension
    U : torch.Tensor
        Tensor of shape (b2, rank, out_f // b2)
    V : torch.Tensor
        Tensor of shape (b1, in_f // b1, rank)
    S : torch.Tensor
        Scaling tensor of shape (b1, b2, rank)

    Returns:
    -------
    out: torch.Tensor
        Output tensor of shape (num_batches, num_seq, out_f), resulting from the BLAST matrix multiplication
    """

    b1, p, r = V.shape
    b2, _, q = U.shape
    assert b1 == b2
    assert b1 != 1 and b2 != 1

    x_shape = x.shape
    x = x.flatten(0, -2)
    x = x.view(-1, b1, x_shape[-1] // b1).transpose(0, 1)

    out = torch.empty(b2, x_shape[0] * x_shape[1], q, device=x.device, dtype=x.dtype)
    out = torch.bmm(_triton_blast_partial_launcher_fp16(x, V, S, config), U, out=out)
    out = out.transpose(0, 1).reshape((x_shape[0], x_shape[1], b2 * q))

    return out

#-----------------------------------
def triton_blast_partial_grouped_fp32(
    x: torch.Tensor,
    U: torch.Tensor, 
    V: torch.Tensor, 
    S: torch.Tensor) -> Tuple[torch.Tensor, ...]:

    """
    Performs BLAST matrix multiplication using triton_blast_partial_grouped_launcher_fp32 and is
    useful for checking correctness of entire BLAST operation

    Parameters:
    ----------
    x : torch.Tensor
        Input tensor of shape (num_batches, num_seq, in_f), where in_f is the last dimension
    U : torch.Tensor
        Tensor of shape (b2, rank, out_f // b2)
    V : torch.Tensor
        Tensor of shape (b1, in_f // b1, rank)
    S : torch.Tensor
        Scaling tensor of shape (b1, b2, rank)

    Returns:
    -------
    out: torch.Tensor
        Output tensor of shape (num_batches, num_seq, out_f), resulting from the BLAST matrix multiplication
    """

    best_config = getattr(_triton_blast_partial_grouped_kernel_fp16, 'best_config', None)

    if best_config is None:
        default_config = triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_R': 32, 'BLOCK_SIZE_P': 32, 'GROUP_SIZE_N': 8}, num_stages=3, num_warps=4)
        print(f"No autotuned config found. Running with default config: {default_config}")
        print("Ensure to run triton_blast_partial_grouped_fp16 before")
        best_config = default_config

    b1, p, r = V.shape
    b2, _, q = U.shape
    assert b1 == b2
    assert b1 != 1 and b2 != 1

    x_shape = x.shape
    x = x.flatten(0, -2)
    x = x.view(-1, b1, x_shape[-1] // b1).transpose(0, 1)

    z, y = _triton_blast_partial_grouped_launcher_fp32(x, V, S, best_config)

    out = torch.empty(x_shape[0] * x_shape[1], b2, q, device=x.device, dtype=x.dtype).transpose(0, 1)
    out = torch.bmm(z, U, out=out)
    out = out.transpose(0, 1).reshape((x_shape[0], x_shape[1], b2 * q))

    return out, z, y

def triton_blast_partial_grouped_fp16(
    x: torch.Tensor, 
    U: torch.Tensor, 
    V: torch.Tensor, 
    S: torch.Tensor,
    config: triton.Config = None) -> torch.Tensor:

    """
    Performs BLAST matrix multiplication using triton_blast_partial_grouped_launcher_fp16 and is
    useful for checking correctness of entire BLAST operation

    Parameters:
    ----------
    x : torch.Tensor
        Input tensor of shape (num_batches, num_seq, in_f), where in_f is the last dimension
    U : torch.Tensor
        Tensor of shape (b2, rank, out_f // b2)
    V : torch.Tensor
        Tensor of shape (b1, in_f // b1, rank)
    S : torch.Tensor
        Scaling tensor of shape (b1, b2, rank)

    Returns:
    -------
    out: torch.Tensor
        Output tensor of shape (num_batches, num_seq, out_f), resulting from the BLAST matrix multiplication
    """

    b1, p, r = V.shape
    b2, _, q = U.shape
    assert b1 == b2
    assert b1 != 1 and b2 != 1

    x_shape = x.shape
    x = x.flatten(0, -2)
    x = x.view(-1, b1, x_shape[-1] // b1).transpose(0, 1)

    out = torch.empty(b2, x_shape[0] * x_shape[1], q, device=x.device, dtype=x.dtype)
    out = torch.bmm(_triton_blast_partial_grouped_launcher_fp16(x, V, S, config), U, out=out)
    out = out.transpose(0, 1).reshape((x_shape[0], x_shape[1], b2 * q))

    return out

#-----------------------------------
def triton_blast_partial_grouped_persistent_fp32(
    x: torch.Tensor, 
    U: torch.Tensor, 
    V: torch.Tensor, 
    S: torch.Tensor) -> Tuple[torch.Tensor, ...]:

    """
    Performs BLAST matrix multiplication using triton_blast_partial_grouped_persistent_launcher_fp32 and is
    useful for checking correctness of entire BLAST operation

    Parameters:
    ----------
    x : torch.Tensor
        Input tensor of shape (num_batches, num_seq, in_f), where in_f is the last dimension
    U : torch.Tensor
        Tensor of shape (b2, rank, out_f // b2)
    V : torch.Tensor
        Tensor of shape (b1, in_f // b1, rank)
    S : torch.Tensor
        Scaling tensor of shape (b1, b2, rank)

    Returns:
    -------
    out: torch.Tensor
        Output tensor of shape (num_batches, num_seq, out_f), resulting from the BLAST matrix multiplication
    """

    best_config = getattr(_triton_blast_partial_grouped_persistent_kernel_fp16, 'best_config', None)

    if best_config is None:
        default_config = triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_R': 32, 'BLOCK_SIZE_P': 32, 'GROUP_SIZE_N': 8}, num_stages=3, num_warps=4)
        print(f"No autotuned config found. Running with default config: {default_config}")
        print("Ensure to run triton_blast_partial_grouped_persistent_fp16 before")
        best_config = default_config

    b1, p, r = V.shape
    b2, _, q = U.shape
    assert b1 == b2
    assert b1 != 1 and b2 != 1

    x_shape = x.shape
    x = x.flatten(0, -2)
    x = x.view(-1, b1, x_shape[-1] // b1).transpose(0, 1)

    z, y = _triton_blast_partial_grouped_persistent_launcher_fp32(x, V, S, best_config)

    out = torch.empty(x_shape[0] * x_shape[1], b2, q, device=x.device, dtype=x.dtype).transpose(0, 1)
    out = torch.bmm(z, U, out=out)
    out = out.transpose(0, 1).reshape((x_shape[0], x_shape[1], b2 * q))

    return out, z, y

def triton_blast_partial_grouped_persistent_fp16(
    x: torch.Tensor, 
    U: torch.Tensor, 
    V: torch.Tensor, 
    S: torch.Tensor,
    config: triton.Config = None) -> torch.Tensor:

    """
    Performs BLAST matrix multiplication using triton_blast_partial_grouped_persistent_launcher_fp16 and is
    useful for checking correctness of entire BLAST operation

    Parameters:
    ----------
    x : torch.Tensor
        Input tensor of shape (num_batches, num_seq, in_f), where in_f is the last dimension
    U : torch.Tensor
        Tensor of shape (b2, rank, out_f // b2)
    V : torch.Tensor
        Tensor of shape (b1, in_f // b1, rank)
    S : torch.Tensor
        Scaling tensor of shape (b1, b2, rank)

    Returns:
    -------
    out: torch.Tensor
        Output tensor of shape (num_batches, num_seq, out_f), resulting from the BLAST matrix multiplication
    """

    b1, p, r = V.shape
    b2, _, q = U.shape
    assert b1 == b2
    assert b1 != 1 and b2 != 1

    x_shape = x.shape
    x = x.flatten(0, -2)
    x = x.view(-1, b1, x_shape[-1] // b1).transpose(0, 1)

    out = torch.empty(b2, x_shape[0] * x_shape[1], q, device=x.device, dtype=x.dtype)
    out = torch.bmm(_triton_blast_partial_grouped_persistent_launcher_fp16(x, V, S, config), U, out=out)
    out = out.transpose(0, 1).reshape((x_shape[0], x_shape[1], b2 * q))

    return out

#-----------------------------------
def triton_blast_full_fp32(
    x: torch.Tensor, 
    U: torch.Tensor, 
    V: torch.Tensor, 
    S: torch.Tensor) -> Tuple[torch.Tensor, ...]:

    """
    Performs BLAST matrix multiplication using triton_blast_full_launcher_fp32 and is
    useful for checking correctness of entire BLAST operation

    Parameters:
    ----------
    x : torch.Tensor
        Input tensor of shape (num_batches, num_seq, in_f)
    U : torch.Tensor
        Tensor of shape (b2, rank, out_f // b2)
    V : torch.Tensor
        Tensor of shape (b1, in_f // b1, rank)
    S : torch.Tensor
        Scaling tensor of shape (b1, b2, rank)

    Returns:
    -------
    out: torch.Tensor
        Output tensor of shape (num_batches, num_seq, out_f)
    z: torch.Tensor
        Intermediate output tensor of shape (b2, num_batches * num_seq, rank)
    y: torch.Tensor
        Intermediate output tensor of shape (b1, num_batches * num_seq, rank)
    """

    best_config = getattr(_triton_blast_full_kernel_fp16, 'best_config', None)

    if best_config is None:
        default_config = triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_Q': 32, 'BLOCK_SIZE_P': 32}, num_stages=3, num_warps=4)
        print(f"No autotuned config found. Running with default config: {default_config}")
        print("Ensure to run triton_blast_full_fp16 before")
        best_config = default_config

    b1, p, r = V.shape
    b2, _, q = U.shape
    assert b1 == b2
    assert b1 != 1 and b2 != 1

    x_shape = x.shape
    x = x.flatten(0, -2)
    x = x.view(-1, b1, x_shape[-1] // b1).transpose(0, 1)

    out, z, y = _triton_blast_full_launcher_fp32(x, V, S, U, best_config)
    out = out.transpose(0, 1).reshape((x_shape[0], x_shape[1], b2 * q))

    return out, z, y

def triton_blast_full_fp16(
    x: torch.Tensor, 
    U: torch.Tensor, 
    V: torch.Tensor, 
    S: torch.Tensor,
    config: triton.Config = None) -> torch.Tensor:

    """
    Performs BLAST matrix multiplication using triton_blast_full_launcher_fp16 and is
    useful for checking correctness of entire BLAST operation

    Parameters:
    ----------
    x : torch.Tensor
        Input tensor of shape (num_batches, num_seq, in_f)
    U : torch.Tensor
        Tensor of shape (b2, rank, out_f // b2)
    V : torch.Tensor
        Tensor of shape (b1, in_f // b1, rank)
    S : torch.Tensor
        Scaling tensor of shape (b1, b2, rank)

    Returns:
    -------
    out: torch.Tensor
        Output tensor of shape (num_batches, num_seq, out_f)
    """
    
    b1, p, r = V.shape
    b2, _, q = U.shape
    assert b1 == b2
    assert b1 != 1 and b2 != 1

    x_shape = x.shape
    x = x.flatten(0, -2)
    x = x.view(-1, b1, x_shape[-1] // b1).transpose(0, 1)

    out = _triton_blast_full_launcher_fp16(x, V, S, U, config)
    out = out.transpose(0, 1).reshape((x_shape[0], x_shape[1], b2 * q))

    return out

#-----------------------------------
def triton_blast_bmm_fp32(
    x: torch.Tensor, 
    U: torch.Tensor,
    V: torch.Tensor,
    S: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    
    best_config_xv = getattr(_triton_blast_bmm_xv_kernel_fp16, 'best_config', None)
    best_config_sxv = getattr(_triton_blast_bmm_sxv_kernel_fp16, 'best_config', None)
    best_config_usxv = getattr(_triton_blast_bmm_usxv_kernel_fp16, 'best_config', None)

    if best_config_xv is None:
        default_config = triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_R': 32,  'BLOCK_SIZE_P': 32,  'GROUP_SIZE_N': 8}, num_stages=3, num_warps=4),
        print(f"No autotuned config found. Running with default config: {default_config}")
        print("Ensure to run triton_blast_bmm_usxv_fp16 before")
        best_config_xv = default_config

    if best_config_sxv is None:
        default_config =  triton.Config({'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4)
        print(f"No autotuned config found. Running with default config: {default_config}")
        print("Ensure to run triton_blast_bmm_usxv_fp16 before")
        best_config_sxv = default_config

    if best_config_usxv is None:
        default_config = triton.Config({'BLOCK_SIZE_Q': 64,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_R': 32,  'GROUP_SIZE_Q': 8}, num_stages=3, num_warps=4),
        print(f"No autotuned config found. Running with default config: {default_config}")
        print("Ensure to run triton_blast_bmm_usxv_fp16 before")
        best_config_usxv = default_config

    b1, p, r = V.shape
    b2, q, _ = U.shape

    assert b1 == b2
    assert b1 != 1 and b2 != 1

    x_shape = x.shape
    x = x.flatten(0, -2)
    x = x.view(-1, b1, x_shape[-1] // b1).transpose(0, 1)

    y = _triton_blast_bmm_xv_launcher_fp32(x, V, best_config_xv)
    z = _triton_blast_bmm_sxv_launcher_fp32(y, S, best_config_sxv)
    out = _triton_blast_bmm_usxv_launcher_fp32(z, U, best_config_usxv)
    out = out.reshape((x_shape[0], x_shape[1], b2 * q))
    
    return out, z, y

def triton_blast_bmm_fp16(
    x: torch.Tensor, 
    U: torch.Tensor,
    V: torch.Tensor,
    S: torch.Tensor,
    config: Tuple[triton.Config, ...] = (None, None, None)) -> torch.Tensor:
    
    b1, p, r = V.shape
    b2, q, _ = U.shape

    assert b1 == b2
    assert b1 != 1 and b2 != 1

    x_shape = x.shape
    x = x.flatten(0, -2)
    x = x.view(-1, b1, x_shape[-1] // b1).transpose(0, 1)

    y = _triton_blast_bmm_xv_launcher_fp16(x, V, config[0])
    z = _triton_blast_bmm_sxv_launcher_fp16(y, S, config[1])
    out = _triton_blast_bmm_usxv_launcher_fp16(z, U, config[2])
    out = out.reshape((x_shape[0], x_shape[1], b2 * q))

    return out

""" Get Triton BLAST Kernel Autotuned Configuration """
def get_triton_blast_partial_fp16_config():
    return getattr(_triton_blast_partial_kernel_fp16, 'best_config', None)

def get_triton_blast_partial_grouped_fp16_config():
    return getattr(_triton_blast_partial_grouped_kernel_fp16, 'best_config', None)

def get_triton_blast_partial_grouped_persistent_fp16_config():
    return getattr(_triton_blast_partial_grouped_persistent_kernel_fp16, 'best_config', None)

def get_triton_blast_full_fp16_config():
    return getattr(_triton_blast_full_kernel_fp16, 'best_config', None)

def get_triton_blast_bmm_xv_fp16_config():
    return getattr(_triton_blast_bmm_xv_kernel_fp16, 'best_config', None)

def get_triton_blast_bmm_sxv_fp16_config():
    return getattr(_triton_blast_bmm_sxv_kernel_fp16, 'best_config', None)

def get_triton_blast_bmm_usxv_fp16_config():
    return getattr(_triton_blast_bmm_usxv_kernel_fp16, 'best_config', None)