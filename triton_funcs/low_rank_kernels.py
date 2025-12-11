import torch
import triton
import triton.language as tl
from utils import next_power_of_2

""" Autotune Configurations for Triton Low-Rank Kernels """
def _get_triton_low_rank_kernel_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_O': 256, 'BLOCK_SIZE_I': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_O': 256, 'BLOCK_SIZE_I': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_O': 64,  'BLOCK_SIZE_I': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_O': 32,  'BLOCK_SIZE_I': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_O': 256, 'BLOCK_SIZE_I': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_O': 32,  'BLOCK_SIZE_I': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_O': 32,  'BLOCK_SIZE_I': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_O': 64,  'BLOCK_SIZE_I': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_O': 64,  'BLOCK_SIZE_I': 32}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_O': 64,  'BLOCK_SIZE_I': 32}, num_stages=5, num_warps=2)
    ]

""" Triton Low-Rank Kernels """
@triton.jit
def _triton_low_rank_kernel_fp32(
    x_ptr, a_ptr, b_ptr, y_ptr, z_ptr,
    N, O, I, R,
    stride_xn, stride_xi,
    stride_ai, stride_ar,
    stride_br, stride_bo,
    stride_yn, stride_yo,
    stride_zn, stride_zr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_R: tl.constexpr
    ):
    
    pid_n = tl.program_id(axis=0)
    offs_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    offs_r = tl.arange(0, BLOCK_SIZE_R)
    x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_i[None, :] * stride_xi)
    a_ptrs = a_ptr + (offs_i[:, None] * stride_ai + offs_r[None, :] * stride_ar)
    accumulator_in = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)

    for i in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
        x_mask = ((offs_i[None, :] < (I - i * BLOCK_SIZE_I)) & (offs_xn[:, None] < N))
        a_mask = ((offs_i[:, None] < (I - i * BLOCK_SIZE_I)) & (offs_r[None, :] < R))
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        accumulator_in = tl.dot(x, a, accumulator_in, allow_tf32=False)
        x_ptrs += BLOCK_SIZE_I * stride_xi
        a_ptrs += BLOCK_SIZE_I * stride_ai

    offs_zn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    z_ptrs = z_ptr + stride_zn * offs_zn[:, None] + stride_zr * offs_r[None, :]
    z_mask = (offs_zn[:, None] < N) & (offs_r[None, :] < R)
    z = accumulator_in
    tl.store(z_ptrs, z, mask=z_mask)

    offs_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_o = tl.arange(0, BLOCK_SIZE_O)
    y_ptrs = y_ptr + (stride_yn * offs_yn[:, None] + stride_yo * offs_o[None, :])
    b_ptrs = b_ptr + (offs_r[:, None] * stride_br + offs_o[None,:] * stride_bo)
    for o in range(0, tl.cdiv(O, BLOCK_SIZE_O)):
        b_mask = ((offs_r[:, None] < R) & (offs_o[None, :] < (O - o * BLOCK_SIZE_O)))
        y_mask = ((offs_yn[:, None] < N) & (offs_o[None, :] < (O - o * BLOCK_SIZE_O)))
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator_out = tl.dot(z, b, allow_tf32=False)
        y = accumulator_out
        tl.store(y_ptrs, y, mask=y_mask)
        y_ptrs += BLOCK_SIZE_O * stride_yo
        b_ptrs += BLOCK_SIZE_O * stride_bo

@triton.autotune(configs=_get_triton_low_rank_kernel_autotune_config(), key=['N', 'O', 'I', 'R'])
@triton.jit
def _triton_low_rank_kernel_fp16(
    x_ptr, a_ptr, b_ptr, y_ptr,
    N, O, I, R,
    stride_xn, stride_xi,
    stride_ai, stride_ar,
    stride_br, stride_bo,
    stride_yn, stride_yo,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_R: tl.constexpr
    ):
    
    pid_n = tl.program_id(axis=0)
    offs_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    offs_r = tl.arange(0, BLOCK_SIZE_R)
    x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_i[None, :] * stride_xi)
    a_ptrs = a_ptr + (offs_i[:, None] * stride_ai + offs_r[None, :] * stride_ar)
    accumulator_in = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)

    for i in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
        x_mask = ((offs_i[None, :] < (I - i * BLOCK_SIZE_I)) & (offs_xn[:, None] < N))
        a_mask = ((offs_i[:, None] < (I - i * BLOCK_SIZE_I)) & (offs_r[None, :] < R))
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        accumulator_in = tl.dot(x, a, accumulator_in)
        x_ptrs += BLOCK_SIZE_I * stride_xi
        a_ptrs += BLOCK_SIZE_I * stride_ai

    z = accumulator_in.to(tl.bfloat16)

    offs_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_o = tl.arange(0, BLOCK_SIZE_O)
    y_ptrs = y_ptr + (stride_yn * offs_yn[:, None] + stride_yo * offs_o[None, :])
    b_ptrs = b_ptr + (offs_r[:, None] * stride_br + offs_o[None,:] * stride_bo)
    for o in range(0, tl.cdiv(O, BLOCK_SIZE_O)):
        b_mask = ((offs_r[:, None] < R) & (offs_o[None, :] < (O - o * BLOCK_SIZE_O)))
        y_mask = ((offs_yn[:, None] < N) & (offs_o[None, :] < (O - o * BLOCK_SIZE_O)))
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator_out = tl.dot(z, b)
        y = accumulator_out.to(tl.bfloat16)
        tl.store(y_ptrs, y, mask=y_mask)
        y_ptrs += BLOCK_SIZE_O * stride_yo
        b_ptrs += BLOCK_SIZE_O * stride_bo

@triton.jit
def _triton_low_rank_kernel_fp16_no_autotune(
    x_ptr, a_ptr, b_ptr, y_ptr,
    N, O, I, R,
    stride_xn, stride_xi,
    stride_ai, stride_ar,
    stride_br, stride_bo,
    stride_yn, stride_yo,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_R: tl.constexpr
    ):
    
    pid_n = tl.program_id(axis=0)
    offs_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    offs_r = tl.arange(0, BLOCK_SIZE_R)
    x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_i[None, :] * stride_xi)
    a_ptrs = a_ptr + (offs_i[:, None] * stride_ai + offs_r[None, :] * stride_ar)
    accumulator_in = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_R), dtype=tl.float32)

    for i in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
        x_mask = ((offs_i[None, :] < (I - i * BLOCK_SIZE_I)) & (offs_xn[:, None] < N))
        a_mask = ((offs_i[:, None] < (I - i * BLOCK_SIZE_I)) & (offs_r[None, :] < R))
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        accumulator_in = tl.dot(x, a, accumulator_in)
        x_ptrs += BLOCK_SIZE_I * stride_xi
        a_ptrs += BLOCK_SIZE_I * stride_ai

    z = accumulator_in.to(tl.bfloat16)

    offs_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_o = tl.arange(0, BLOCK_SIZE_O)
    y_ptrs = y_ptr + (stride_yn * offs_yn[:, None] + stride_yo * offs_o[None, :])
    b_ptrs = b_ptr + (offs_r[:, None] * stride_br + offs_o[None,:] * stride_bo)
    for o in range(0, tl.cdiv(O, BLOCK_SIZE_O)):
        b_mask = ((offs_r[:, None] < R) & (offs_o[None, :] < (O - o * BLOCK_SIZE_O)))
        y_mask = ((offs_yn[:, None] < N) & (offs_o[None, :] < (O - o * BLOCK_SIZE_O)))
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator_out = tl.dot(z, b)
        y = accumulator_out.to(tl.bfloat16)
        tl.store(y_ptrs, y, mask=y_mask)
        y_ptrs += BLOCK_SIZE_O * stride_yo
        b_ptrs += BLOCK_SIZE_O * stride_bo

@triton.jit
def _triton_low_rank_load_store_kernel_fp32(
    x_ptr, a_ptr, b_ptr, y_ptr,
    N, O, I, R,
    stride_xn, stride_xi,
    stride_ai, stride_ar,
    stride_br, stride_bo,
    stride_yn, stride_yo,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_R: tl.constexpr
    ):
    
    pid_n = tl.program_id(axis=0)
    offs_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    offs_r = tl.arange(0, BLOCK_SIZE_R)
    x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_i[None, :] * stride_xi)
    a_ptrs = a_ptr + (offs_i[:, None] * stride_ai + offs_r[None, :] * stride_ar)
    for i in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
        x_mask = ((offs_i[None, :] < (I - i * BLOCK_SIZE_I)) & (offs_xn[:, None] < N))
        a_mask = ((offs_i[:, None] < (I - i * BLOCK_SIZE_I)) & (offs_r[None, :] < R))
        x = tl.load(x_ptrs, mask=x_mask, other=0.0, volatile=True)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0, volatile=True)
        x_ptrs += BLOCK_SIZE_I * stride_xi
        a_ptrs += BLOCK_SIZE_I * stride_ai

    offs_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_o = tl.arange(0, BLOCK_SIZE_O)
    y_ptrs = y_ptr + (stride_yn * offs_yn[:, None] + stride_yo * offs_o[None, :])
    b_ptrs = b_ptr + (offs_r[:, None] * stride_br + offs_o[None,:] * stride_bo)
    for o in range(0, tl.cdiv(O, BLOCK_SIZE_O)):
        y = tl.full((BLOCK_SIZE_N, BLOCK_SIZE_O), o, dtype=tl.float32)
        b_mask = ((offs_r[:, None] < R) & (offs_o[None, :] < (O - o * BLOCK_SIZE_O)))
        y_mask = ((offs_yn[:, None] < N) & (offs_o[None, :] < (O - o * BLOCK_SIZE_O)))
        b = tl.load(b_ptrs, mask=b_mask, other=0.0, volatile=True)
        tl.store(y_ptrs, y, mask=y_mask)
        y_ptrs += BLOCK_SIZE_O * stride_yo
        b_ptrs += BLOCK_SIZE_O * stride_bo

@triton.jit
def _triton_low_rank_load_store_kernel_fp16(
    x_ptr, a_ptr, b_ptr, y_ptr,
    N, O, I, R,
    stride_xn, stride_xi,
    stride_ai, stride_ar,
    stride_br, stride_bo,
    stride_yn, stride_yo,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_R: tl.constexpr
    ):
    
    pid_n = tl.program_id(axis=0)
    offs_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    offs_r = tl.arange(0, BLOCK_SIZE_R)
    x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_i[None, :] * stride_xi)
    a_ptrs = a_ptr + (offs_i[:, None] * stride_ai + offs_r[None, :] * stride_ar)
    for i in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
        x_mask = ((offs_i[None, :] < (I - i * BLOCK_SIZE_I)) & (offs_xn[:, None] < N))
        a_mask = ((offs_i[:, None] < (I - i * BLOCK_SIZE_I)) & (offs_r[None, :] < R))
        x = tl.load(x_ptrs, mask=x_mask, other=0.0, volatile=True)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0, volatile=True)
        x_ptrs += BLOCK_SIZE_I * stride_xi
        a_ptrs += BLOCK_SIZE_I * stride_ai

    offs_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_o = tl.arange(0, BLOCK_SIZE_O)
    y_ptrs = y_ptr + (stride_yn * offs_yn[:, None] + stride_yo * offs_o[None, :])
    b_ptrs = b_ptr + (offs_r[:, None] * stride_br + offs_o[None,:] * stride_bo)
    for o in range(0, tl.cdiv(O, BLOCK_SIZE_O)):
        y = tl.full((BLOCK_SIZE_N, BLOCK_SIZE_O), o, dtype=tl.bfloat16)
        b_mask = ((offs_r[:, None] < R) & (offs_o[None, :] < (O - o * BLOCK_SIZE_O)))
        y_mask = ((offs_yn[:, None] < N) & (offs_o[None, :] < (O - o * BLOCK_SIZE_O)))
        b = tl.load(b_ptrs, mask=b_mask, other=0.0, volatile=True)
        tl.store(y_ptrs, y, mask=y_mask)
        y_ptrs += BLOCK_SIZE_O * stride_yo
        b_ptrs += BLOCK_SIZE_O * stride_bo

""" Triton Low-Rank Kernel Launchers """
def _triton_low_rank_launcher_fp32(
    x: torch.Tensor, 
    a: torch.Tensor, 
    b: torch.Tensor,
    best_config: triton.Config) -> torch.Tensor:

    assert x.shape[1] == a.shape[0] and a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert x.is_contiguous(), "Matrix X must be contiguous"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    assert x.dtype == torch.float32
    assert a.dtype == torch.float32
    assert b.dtype == torch.float32

    N, I = x.shape
    I, R = a.shape
    R, O = b.shape
    y = torch.empty((N, O), device=x.device, dtype=x.dtype)
    z = torch.empty((N, R), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(N, best_config.kwargs['BLOCK_SIZE_N']), )
    _triton_low_rank_kernel_fp32[grid](
        x, a, b, y, z,
        N, O, I, R,
        x.stride(0), x.stride(1),
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        y.stride(0), y.stride(1),
        z.stride(0), z.stride(1),
        BLOCK_SIZE_N=best_config.kwargs['BLOCK_SIZE_N'],
        BLOCK_SIZE_O=best_config.kwargs['BLOCK_SIZE_O'],
        BLOCK_SIZE_I=best_config.kwargs['BLOCK_SIZE_I'],
        BLOCK_SIZE_R=next_power_of_2(R),
        num_warps=best_config.num_warps,
        num_stages=1        
    )
    return y, z

def _triton_low_rank_launcher_fp16(
    x: torch.Tensor, 
    a: torch.Tensor, 
    b: torch.Tensor,
    config:triton.Config = None) -> torch.Tensor:

    assert x.shape[1] == a.shape[0] and a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert x.is_contiguous(), "Matrix X must be contiguous"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    assert x.dtype == torch.bfloat16
    assert a.dtype == torch.bfloat16
    assert b.dtype == torch.bfloat16

    N, I = x.shape
    I, R = a.shape
    R, O = b.shape
    y = torch.empty((N, O), device=x.device, dtype=x.dtype)
    if config is None:
        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']), )
        _triton_low_rank_kernel_fp16[grid](
            x, a, b, y,
            N, O, I, R,
            x.stride(0), x.stride(1),
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_SIZE_R=next_power_of_2(R)
        )
    else:
        grid = (triton.cdiv(N, config.kwargs['BLOCK_SIZE_N']), )
        _triton_low_rank_kernel_fp16_no_autotune[grid](
            x, a, b, y,
            N, O, I, R,
            x.stride(0), x.stride(1),
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_SIZE_N=config.kwargs['BLOCK_SIZE_N'],
            BLOCK_SIZE_O=config.kwargs['BLOCK_SIZE_O'],
            BLOCK_SIZE_I=config.kwargs['BLOCK_SIZE_I'],
            BLOCK_SIZE_R=next_power_of_2(R),
            num_warps=config.num_warps,
            num_stages=config.num_stages   
        )
    return y

def _triton_low_rank_load_store_launcher_fp32(
    x: torch.Tensor, 
    a: torch.Tensor, 
    b: torch.Tensor, 
    best_config: triton.Config) -> torch.Tensor:

    assert x.shape[1] == a.shape[0] and a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert x.is_contiguous(), "Matrix X must be contiguous"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    assert x.dtype == torch.float32
    assert a.dtype == torch.float32
    assert b.dtype == torch.float32
    
    N, I = x.shape
    I, R = a.shape
    R, O = b.shape
    y = torch.empty((N, O), device=x.device, dtype=x.dtype)

    _triton_low_rank_load_store_kernel_fp32[(triton.cdiv(N, best_config.kwargs['BLOCK_SIZE_N']), )](
        x, a, b, y,
        N, O, I, R,
        x.stride(0), x.stride(1),
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_SIZE_N=best_config.kwargs['BLOCK_SIZE_N'],
        BLOCK_SIZE_O=best_config.kwargs['BLOCK_SIZE_O'],
        BLOCK_SIZE_I=best_config.kwargs['BLOCK_SIZE_I'],
        BLOCK_SIZE_R=next_power_of_2(R),
        num_warps=best_config.num_warps,
        num_stages=best_config.num_stages
    )
    return y

def _triton_low_rank_load_store_launcher_fp16(
    x: torch.Tensor, 
    a: torch.Tensor, 
    b: torch.Tensor, 
    best_config: triton.Config) -> torch.Tensor:

    assert x.shape[1] == a.shape[0] and a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert x.is_contiguous(), "Matrix X must be contiguous"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    assert x.dtype == torch.bfloat16
    assert a.dtype == torch.bfloat16
    assert b.dtype == torch.bfloat16
    
    N, I = x.shape
    I, R = a.shape
    R, O = b.shape
    y = torch.empty((N, O), device=x.device, dtype=x.dtype)

    _triton_low_rank_load_store_kernel_fp16[(triton.cdiv(N, best_config.kwargs['BLOCK_SIZE_N']), )](
        x, a, b, y,
        N, O, I, R,
        x.stride(0), x.stride(1),
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_SIZE_N=best_config.kwargs['BLOCK_SIZE_N'],
        BLOCK_SIZE_O=best_config.kwargs['BLOCK_SIZE_O'],
        BLOCK_SIZE_I=best_config.kwargs['BLOCK_SIZE_I'],
        BLOCK_SIZE_R=next_power_of_2(R),
        num_warps=best_config.num_warps,
        num_stages=best_config.num_stages
    )
    return y

""" Triton Low-Rank Functions """
def triton_low_rank_fp32(
    x: torch.Tensor, 
    a: torch.Tensor, 
    b: torch.Tensor) -> torch.Tensor:

    best_config = getattr(_triton_low_rank_kernel_fp16, 'best_config', None)

    if best_config is None:
        default_config = triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32}, num_stages=3, num_warps=4)
        print(f"No autotuned config found. Running with default config: {default_config}")
        print("Ensure to run triton_low_rank_fp16 before")
        best_config = default_config

    x = x.flatten(0, -2)
    return _triton_low_rank_launcher_fp32(x, a, b, best_config)

def triton_low_rank_fp16(
    x: torch.Tensor, 
    a: torch.Tensor, 
    b: torch.Tensor,
    config: triton.Config = None) -> torch.Tensor:

    x = x.flatten(0, -2)
    return _triton_low_rank_launcher_fp16(x, a, b, config)

def triton_low_rank_load_store_fp32(
    x: torch.Tensor, 
    a: torch.Tensor, 
    b: torch.Tensor) -> torch.Tensor:

    best_config = getattr(_triton_low_rank_kernel_fp32, 'best_config', None)

    if best_config is None:
        default_config = triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32}, num_stages=3, num_warps=4)
        print(f"No autotuned config found. Running with default config: {default_config}")
        print("Ensure to run triton_low_rank_fp32 before")
        best_config = default_config

    x = x.flatten(0, -2)
    return _triton_low_rank_load_store_launcher_fp32(x, a, b, best_config)

def triton_low_rank_load_store_fp16(
    x: torch.Tensor, 
    a: torch.Tensor, 
    b: torch.Tensor) -> torch.Tensor:

    best_config = getattr(_triton_low_rank_kernel_fp16, 'best_config', None)

    if best_config is None:
        default_config = triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32}, num_stages=3, num_warps=4)
        print(f"No autotuned config found. Running with default config: {default_config}")
        print("Ensure to run triton_low_rank_fp16 before")
        best_config = default_config
    
    x = x.flatten(0, -2)
    return _triton_low_rank_load_store_launcher_fp16(x, a, b, best_config)

""" Get Triton Low-Rank Kernel Autotuned Configuration """
def get_triton_low_rank_fp16_config():
    return getattr(_triton_low_rank_kernel_fp16, 'best_config', None)