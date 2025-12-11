import triton
import triton.language as tl
from triton.language.extra import libdevice
import torch
from utils import next_power_of_2
from typing import Tuple

""" Autotune Configurations for Triton BLAST Sym Quant Kernels """
def _get_triton_blast_bmm_xv_int8_kernel_autotune_config():
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

def _get_triton_blast_bmm_sxv_int8_kernel_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_N': BLOCK_SIZE_N}, num_stages=num_stages, num_warps=num_warps)
        for BLOCK_SIZE_N in [32, 64, 128, 256, 512]
        for num_stages in [1, 2, 3, 4, 5, 6]
        for num_warps in [2, 4, 8]
    ]

def _get_triton_blast_bmm_usxv_int8_kernel_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_Q': BLOCK_SIZE_Q, 'BLOCK_SIZE_N': BLOCK_SIZE_N, 'BLOCK_SIZE_R': 32, 'GROUP_SIZE_Q': GROUP_SIZE_Q}, num_stages=num_stages, num_warps=num_warps)
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_N in [64, 128]
        for GROUP_SIZE_Q in [4, 8]
        for num_stages in [3, 4, 5]
        for num_warps in [4, 8]
    ]

""" Triton BLAST Sym Quant Kernels """
#-----------------------------------
@triton.autotune(configs=_get_triton_blast_bmm_xv_int8_kernel_autotune_config(), key=['N', 'P', 'R', 'B1'])
@triton.jit
def _triton_blast_bmm_xv_kernel_int8_fp16(
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
    result in Y. The computation is done in float32 accumulation and the result is rounded, clamped, and 
    downcast to int8.
    
    Parameters:
    -----------
    x_ptr : tl.pointer
        Pointer to the input tensor X of shape (B1, N, P), in BF16.
    v_ptr : tl.pointer
        Pointer to the input tensor V of shape (B1, P, R), in BF16.
    y_ptr : tl.pointer
        Pointer to the output tensor Y of shape (B1, R, N), in INT8.
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
    
    y = accumulator_in.trans()
    y = libdevice.nearbyint(y)
    y = tl.clamp(y, -128, 127)
    y = y.to(tl.int8)

    offs_yn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_yr = (pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R))
    y_ptrs = y_ptr + offs_yn[None, :] * stride_yn + offs_yr[:, None] * stride_yr + b1 * stride_yb1
    y_mask = (offs_yn[None, :] < N) & (offs_yr[:, None] < R)
    tl.store(y_ptrs, y, mask=y_mask)

@triton.jit
def _triton_blast_bmm_xv_kernel_int8_fp16_no_autotune(
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
@triton.autotune(configs=_get_triton_blast_bmm_sxv_int8_kernel_autotune_config(), key=['N', 'B2', 'R', 'B1'])
@triton.jit
def _triton_blast_bmm_sxv_kernel_int8_fp16(
    y_ptr, s_ptr, z_ptr,
    N, B1, B2, R,
    stride_sr, stride_sb2, stride_sb1,
    stride_yr, stride_yb1, stride_yn,
    stride_zb2, stride_zr, stride_zn,
    BLOCK_SIZE_B2: tl.constexpr, BLOCK_SIZE_B1: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):

    """
    Triton kernel for batched matrix multiplication: Z[b2, r, n] = S[r, b2, b1] @ Y[r, b1, n]

    This kernel computes a batched matrix multiplication where each output tile Z[b2, r, n]
    is the result of multiplying a tensor S[r, b2, b1] with a tensor Y[r, b1, n]. The computation 
    is done in float32 accumulation after upcasting Y to bfloat16 and the result is rounded, 
    clamped, and downcast to int8.

    Parameters:
    -----------
    y_ptr : tl.pointer
        Pointer to the input tensor Y of shape (R, B1, N), stored in INT8.
    s_ptr : tl.pointer
        Pointer to the input tensor S of shape (R, B2, B1), stored in BF16.
    z_ptr : tl.pointer
        Pointer to the output tensor Z of shape (B2, R, N), stored in INT8.

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

    y = y.to(tl.bfloat16)
    accumulator_in = tl.dot(s, y, accumulator_in)

    z = accumulator_in
    z = libdevice.nearbyint(z)
    z = tl.clamp(z, -128, 127)
    z = z.to(tl.int8)

    offs_zn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_zb2 = (tl.arange(0, BLOCK_SIZE_B2))
    z_ptrs = z_ptr + offs_zn[None, :] * stride_zn + offs_zb2[:, None] * stride_zb2 + pid_r * stride_zr
    z_mask = (offs_zn[None, :] < N) & (offs_zb2[:, None] < B2)
    tl.store(z_ptrs, z, mask=z_mask)

@triton.jit
def _triton_blast_bmm_sxv_kernel_int8_fp16_no_autotune(
    y_ptr, s_ptr, z_ptr,
    N, B1, B2, R,
    stride_sr, stride_sb2, stride_sb1,
    stride_yr, stride_yb1, stride_yn,
    stride_zb2, stride_zr, stride_zn,
    BLOCK_SIZE_B2: tl.constexpr, BLOCK_SIZE_B1: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):

    """
    Triton kernel for batched matrix multiplication: Z[b2, r, n] = S[r, b2, b1] @ Y[r, b1, n]

    This kernel computes a batched matrix multiplication where each output tile Z[b2, r, n]
    is the result of multiplying a tensor S[r, b2, b1] with a tensor Y[r, b1, n]. The computation 
    is done in float32 accumulation after upcasting Y to bfloat16 and the result is rounded, 
    clamped, and downcast to int8.

    Parameters:
    -----------
    y_ptr : tl.pointer
        Pointer to the input tensor Y of shape (R, B1, N), stored in INT8.
    s_ptr : tl.pointer
        Pointer to the input tensor S of shape (R, B2, B1), stored in BF16.
    z_ptr : tl.pointer
        Pointer to the output tensor Z of shape (B2, R, N), stored in INT8.

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

    y = y.to(tl.bfloat16)
    accumulator_in = tl.dot(s, y, accumulator_in)

    z = accumulator_in
    z = libdevice.nearbyint(z)
    z = tl.clamp(z, -128, 127)
    z = z.to(tl.int8)

    offs_zn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_zb2 = (tl.arange(0, BLOCK_SIZE_B2))
    z_ptrs = z_ptr + offs_zn[None, :] * stride_zn + offs_zb2[:, None] * stride_zb2 + pid_r * stride_zr
    z_mask = (offs_zn[None, :] < N) & (offs_zb2[:, None] < B2)
    tl.store(z_ptrs, z, mask=z_mask)

#-----------------------------------
@triton.autotune(configs=_get_triton_blast_bmm_usxv_int8_kernel_autotune_config(), key=['N', 'R', 'Q', 'B2'])
@triton.jit
def _triton_blast_bmm_usxv_kernel_int8_fp16(
    z_ptr, u_ptr, out_ptr,
    N, R, B2, Q,
    stride_zb2, stride_zr, stride_zn,
    stride_ub2, stride_uq, stride_ur,
    stride_outn, stride_outb2, stride_outq,
    BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_Q: tl.constexpr):

    """
    Triton kernel for batched matrix multiplication: OUT[b2, q, n] = U[b2, q, r] @ Z[b2, r, n]

    This kernel performs a batched matrix multiplication where a tensor U is multiplied with a
    tensor Z to produce an output tensor OUT. The computation is done in float32 accumulation 
    after upcasting Y to bfloat16 and the result is rounded, clamped, and downcast to int8.

    Parameters:
    -----------
    z_ptr : tl.pointer
        Pointer to tensor Z of shape (B2, R, N), stored in INT8.
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
        z = tl.load(z_ptrs, mask=z_mask, other=0.0)
        z1 = z.to(tl.bfloat16)
        u = tl.load(u_ptrs, mask=u_mask, other=0.0)

        accumulator_in = tl.dot(u, z1, accumulator_in)
        u_ptrs += BLOCK_SIZE_R * stride_ur
        z_ptrs += BLOCK_SIZE_R * stride_zr
    
    out = accumulator_in.to(tl.bfloat16).trans()

    offs_outn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_outq = (pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q))
    out_ptrs = out_ptr + offs_outn[:, None] * stride_outn + offs_outq[None, :] * stride_outq + b2 * stride_outb2
    out_mask = (offs_outn[:, None] < N) & (offs_outq[None, :] < Q)
    tl.store(out_ptrs, out, mask=out_mask)

@triton.jit
def _triton_blast_bmm_usxv_kernel_int8_fp16_no_autotune(
    z_ptr, u_ptr, out_ptr,
    N, R, B2, Q,
    stride_zb2, stride_zr, stride_zn,
    stride_ub2, stride_uq, stride_ur,
    stride_outn, stride_outb2, stride_outq,
    BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_Q: tl.constexpr):

    """
    Triton kernel for batched matrix multiplication: OUT[b2, q, n] = U[b2, q, r] @ Z[b2, r, n]

    This kernel performs a batched matrix multiplication where a tensor U is multiplied with a
    tensor Z to produce an output tensor OUT. The computation is done in float32 accumulation 
    after upcasting Y to bfloat16 and the result is rounded, clamped, and downcast to int8.

    Parameters:
    -----------
    z_ptr : tl.pointer
        Pointer to tensor Z of shape (B2, R, N), stored in INT8.
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
        z = tl.load(z_ptrs, mask=z_mask, other=0.0)
        z1 = z.to(tl.bfloat16)
        u = tl.load(u_ptrs, mask=u_mask, other=0.0)

        accumulator_in = tl.dot(u, z1, accumulator_in)
        u_ptrs += BLOCK_SIZE_R * stride_ur
        z_ptrs += BLOCK_SIZE_R * stride_zr
    
    out = accumulator_in.to(tl.bfloat16).trans()

    offs_outn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_outq = (pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q))
    out_ptrs = out_ptr + offs_outn[:, None] * stride_outn + offs_outq[None, :] * stride_outq + b2 * stride_outb2
    out_mask = (offs_outn[:, None] < N) & (offs_outq[None, :] < Q)
    tl.store(out_ptrs, out, mask=out_mask)


""" Triton BLAST Sym Quant Kernel Launchers """
#-----------------------------------
def _triton_blast_bmm_xv_launcher_int8_fp16(
    x: torch.Tensor, 
    v: torch.Tensor,
    config: triton.Config = None) -> torch.Tensor:
    
    assert x.shape[2] == v.shape[1] and x.shape[0] == v.shape[0], "Incompatible dimensions X and V"
    assert v.is_contiguous(), "Matrix V must be contiguous"
    assert x.dtype == torch.bfloat16
    assert v.dtype == torch.bfloat16

    B1, N, P = x.shape
    B1, P, R = v.shape

    y = torch.empty((R, B1, N), device=x.device, dtype=torch.int8)

    if config is None:
        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']) * triton.cdiv(R, META['BLOCK_SIZE_R']), B1)
        _triton_blast_bmm_xv_kernel_int8_fp16[grid](
            x, v, y,
            N, P, B1, R,
            x.stride(0), x.stride(1), x.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            y.stride(0), y.stride(1), y.stride(2)
        )
    else:
        grid = (triton.cdiv(N, config.kwargs['BLOCK_SIZE_N']) * triton.cdiv(R, config.kwargs['BLOCK_SIZE_R']), B1)
        _triton_blast_bmm_xv_kernel_int8_fp16_no_autotune[grid](
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
def _triton_blast_bmm_sxv_launcher_int8_fp16(
    y: torch.Tensor, 
    s: torch.Tensor,
    config: triton.Config = None) -> torch.Tensor:
    
    assert y.shape[1] == s.shape[2] and y.shape[0] == s.shape[0], "Incompatible dimensions Y and S"
    assert s.is_contiguous(), "Matrix S must be contiguous"
    assert y.dtype == torch.int8
    assert s.dtype == torch.bfloat16

    R, B1, N = y.shape
    R, B2, B1 = s.shape

    z = torch.empty((B2, R, N), device=y.device, dtype=y.dtype)

    BLOCK_SIZE_B1 = max(next_power_of_2(B1), 16)
    BLOCK_SIZE_B2 = max(next_power_of_2(B2), 16)

    if config is None:
        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']), R)
        _triton_blast_bmm_sxv_kernel_int8_fp16[grid](
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
        _triton_blast_bmm_sxv_kernel_int8_fp16_no_autotune[grid](
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
def _triton_blast_bmm_usxv_launcher_int8_fp16(
    z: torch.Tensor,
    u: torch.Tensor,
    config: triton.Config = None) -> torch.Tensor: 
    
    assert u.shape[2] == z.shape[1] and z.shape[0] == u.shape[0], "Incompatible dimensions Z and U"
    assert u.is_contiguous(), "Matrix U must be contiguous"
    assert z.dtype == torch.int8
    assert u.dtype == torch.bfloat16

    B2, R, N = z.shape
    B2, Q, R = u.shape

    out = torch.empty((N, B2, Q), device=z.device, dtype=torch.bfloat16)

    if config is None:
        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']) * triton.cdiv(Q, META['BLOCK_SIZE_Q']), B2)
        _triton_blast_bmm_usxv_kernel_int8_fp16[grid](
            z, u, out,
            N, R, B2, Q,
            z.stride(0), z.stride(1), z.stride(2),
            u.stride(0), u.stride(1), u.stride(2),
            out.stride(0), out.stride(1), out.stride(2)
        )
    else:
        grid = (triton.cdiv(N, config.kwargs['BLOCK_SIZE_N']) * triton.cdiv(Q, config.kwargs['BLOCK_SIZE_Q']), B2)
        _triton_blast_bmm_usxv_kernel_int8_fp16_no_autotune[grid](
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

""" Triton BLAST Sym Quant Functions """
def triton_blast_bmm_int8_fp16(
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

    y = _triton_blast_bmm_xv_launcher_int8_fp16(x, V, config[0])
    z = _triton_blast_bmm_sxv_launcher_int8_fp16(y, S, config[1])
    out = _triton_blast_bmm_usxv_launcher_int8_fp16(z, U, config[2])
    out = out.reshape((x_shape[0], x_shape[1], b2 * q))

    return out

""" Get Triton BLAST Sym Quant Kernel Autotuned Configuration """
def get_triton_blast_bmm_xv_int8_fp16_config():
    return getattr(_triton_blast_bmm_xv_kernel_int8_fp16, 'best_config', None)

def get_triton_blast_bmm_sxv_int8_fp16_config():
    return getattr(_triton_blast_bmm_sxv_kernel_int8_fp16, 'best_config', None)

def get_triton_blast_bmm_usxv_int8_fp16_config():
    return getattr(_triton_blast_bmm_usxv_kernel_int8_fp16, 'best_config', None)