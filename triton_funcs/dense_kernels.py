import torch
import triton
import triton.language as tl

""" Autotune Configurations for Triton Dense Kernels """
def _get_triton_dense_kernel_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_O': 256, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_N': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_N': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_O': 64,  'BLOCK_SIZE_I': 32, 'GROUP_SIZE_N': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_O': 32,  'BLOCK_SIZE_I': 32, 'GROUP_SIZE_N': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_O': 256, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_N': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_N': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_O': 32,  'BLOCK_SIZE_I': 32, 'GROUP_SIZE_N': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_O': 64,  'BLOCK_SIZE_I': 32, 'GROUP_SIZE_N': 8}, num_stages=5, num_warps=2)
    ]

""" Triton Dense Kernels """
@triton.jit
def _triton_dense_kernel_fp32(
    x_ptr, w_ptr, y_ptr,
    N, O, I,
    stride_xn, stride_xi,
    stride_wi, stride_wo,
    stride_yn, stride_yo, 
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr
    ):
    
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N) 
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_O)
    num_pid_in_group = GROUP_SIZE_N * num_pid_o
    group_id = pid // num_pid_in_group
    first_pid_n = group_id * GROUP_SIZE_N
    group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_N)
    pid_n = first_pid_n + ((pid % num_pid_in_group) % group_size_n)
    pid_o = (pid % num_pid_in_group) // group_size_n

    offs_xn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_wo = (pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)) % O
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_i[None, :] * stride_xi)
    w_ptrs = w_ptr + (offs_i[:, None] * stride_wi + offs_wo[None, :] * stride_wo)

    accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_O), dtype=tl.float32)
    for i in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
        x = tl.load(x_ptrs, mask=offs_i[None, :] < (I - i * BLOCK_SIZE_I), other=0.0)
        w = tl.load(w_ptrs, mask=offs_i[:, None] < (I - i * BLOCK_SIZE_I), other=0.0)
        accumulator = tl.dot(x, w, accumulator, allow_tf32 = False)
        x_ptrs += BLOCK_SIZE_I * stride_xi
        w_ptrs += BLOCK_SIZE_I * stride_wi

    y = accumulator

    offs_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_yo = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    y_ptrs = y_ptr + stride_yn * offs_yn[:, None] + stride_yo * offs_yo[None, :]
    y_mask = (offs_yn[:, None] < N) & (offs_yo[None, :] < O)
    tl.store(y_ptrs, y, mask=y_mask)

@triton.autotune(configs=_get_triton_dense_kernel_autotune_config(), key=['N', 'O', 'I'])
@triton.jit
def _triton_dense_kernel_fp16(
    x_ptr, w_ptr, y_ptr,
    N, O, I,
    stride_xn, stride_xi,
    stride_wi, stride_wo,
    stride_yn, stride_yo, 
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr
    ):
    
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N) 
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_O)
    num_pid_in_group = GROUP_SIZE_N * num_pid_o
    group_id = pid // num_pid_in_group
    first_pid_n = group_id * GROUP_SIZE_N
    group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_N)
    pid_n = first_pid_n + ((pid % num_pid_in_group) % group_size_n)
    pid_o = (pid % num_pid_in_group) // group_size_n

    offs_xn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_wo = (pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)) % O
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_i[None, :] * stride_xi)
    w_ptrs = w_ptr + (offs_i[:, None] * stride_wi + offs_wo[None, :] * stride_wo)

    accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_O), dtype=tl.float32)
    for i in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
        x = tl.load(x_ptrs, mask=offs_i[None, :] < (I - i * BLOCK_SIZE_I), other=0.0)
        w = tl.load(w_ptrs, mask=offs_i[:, None] < (I - i * BLOCK_SIZE_I), other=0.0)
        accumulator = tl.dot(x, w, accumulator)
        x_ptrs += BLOCK_SIZE_I * stride_xi
        w_ptrs += BLOCK_SIZE_I * stride_wi

    y = accumulator.to(tl.bfloat16)

    offs_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_yo = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    y_ptrs = y_ptr + stride_yn * offs_yn[:, None] + stride_yo * offs_yo[None, :]
    y_mask = (offs_yn[:, None] < N) & (offs_yo[None, :] < O)
    tl.store(y_ptrs, y, mask=y_mask)

@triton.jit
def _triton_dense_kernel_fp16_no_autotune(
    x_ptr, w_ptr, y_ptr,
    N, O, I,
    stride_xn, stride_xi,
    stride_wi, stride_wo,
    stride_yn, stride_yo, 
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr
    ):
    
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N) 
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_O)
    num_pid_in_group = GROUP_SIZE_N * num_pid_o
    group_id = pid // num_pid_in_group
    first_pid_n = group_id * GROUP_SIZE_N
    group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_N)
    pid_n = first_pid_n + ((pid % num_pid_in_group) % group_size_n)
    pid_o = (pid % num_pid_in_group) // group_size_n

    offs_xn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_wo = (pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)) % O
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_i[None, :] * stride_xi)
    w_ptrs = w_ptr + (offs_i[:, None] * stride_wi + offs_wo[None, :] * stride_wo)

    accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_O), dtype=tl.float32)
    for i in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
        x = tl.load(x_ptrs, mask=offs_i[None, :] < (I - i * BLOCK_SIZE_I), other=0.0)
        w = tl.load(w_ptrs, mask=offs_i[:, None] < (I - i * BLOCK_SIZE_I), other=0.0)
        accumulator = tl.dot(x, w, accumulator)
        x_ptrs += BLOCK_SIZE_I * stride_xi
        w_ptrs += BLOCK_SIZE_I * stride_wi

    y = accumulator.to(tl.bfloat16)

    offs_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_yo = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    y_ptrs = y_ptr + stride_yn * offs_yn[:, None] + stride_yo * offs_yo[None, :]
    y_mask = (offs_yn[:, None] < N) & (offs_yo[None, :] < O)
    tl.store(y_ptrs, y, mask=y_mask)

@triton.jit
def _triton_dense_load_store_kernel_fp32(
    x_ptr, w_ptr, y_ptr,
    N, O, I,
    stride_xn, stride_xi,
    stride_wi, stride_wo,
    stride_yn, stride_yo,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr
    ):
    
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N) 
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_O)
    num_pid_in_group = GROUP_SIZE_N * num_pid_o
    group_id = pid // num_pid_in_group
    first_pid_n = group_id * GROUP_SIZE_N
    group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_N)
    pid_n = first_pid_n + ((pid % num_pid_in_group) % group_size_n)
    pid_o = (pid % num_pid_in_group) // group_size_n

    offs_xn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_wo = (pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)) % O
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_i[None, :] * stride_xi)
    w_ptrs = w_ptr + (offs_i[:, None] * stride_wi + offs_wo[None, :] * stride_wo)

    for i in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
        x = tl.load(x_ptrs, mask=offs_i[None, :] < (I - i * BLOCK_SIZE_I), other=0.0, volatile=True)
        w = tl.load(w_ptrs, mask=offs_i[:, None] < (I - i * BLOCK_SIZE_I), other=0.0, volatile=True)
        x_ptrs += BLOCK_SIZE_I * stride_xi
        w_ptrs += BLOCK_SIZE_I * stride_wi
    
    y = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_O), dtype=tl.float32)
    offs_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_yo = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    y_ptrs = y_ptr + stride_yn * offs_yn[:, None] + stride_yo * offs_yo[None, :]
    y_mask = (offs_yn[:, None] < N) & (offs_yo[None, :] < O)
    tl.store(y_ptrs, y, mask=y_mask)

@triton.jit
def _triton_dense_load_store_kernel_fp16(
    x_ptr, w_ptr, y_ptr,
    N, O, I,
    stride_xn, stride_xi,
    stride_wi, stride_wo,
    stride_yn, stride_yo,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr
    ):
    
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N) 
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_O)
    num_pid_in_group = GROUP_SIZE_N * num_pid_o
    group_id = pid // num_pid_in_group
    first_pid_n = group_id * GROUP_SIZE_N
    group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_N)
    pid_n = first_pid_n + ((pid % num_pid_in_group) % group_size_n)
    pid_o = (pid % num_pid_in_group) // group_size_n

    offs_xn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_wo = (pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)) % O
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    x_ptrs = x_ptr + (offs_xn[:, None] * stride_xn + offs_i[None, :] * stride_xi)
    w_ptrs = w_ptr + (offs_i[:, None] * stride_wi + offs_wo[None, :] * stride_wo)

    for i in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
        x = tl.load(x_ptrs, mask=offs_i[None, :] < (I - i * BLOCK_SIZE_I), other=0.0, volatile=True)
        w = tl.load(w_ptrs, mask=offs_i[:, None] < (I - i * BLOCK_SIZE_I), other=0.0, volatile=True)
        x_ptrs += BLOCK_SIZE_I * stride_xi
        w_ptrs += BLOCK_SIZE_I * stride_wi
    
    y = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_O), dtype=tl.bfloat16)
    offs_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_yo = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    y_ptrs = y_ptr + stride_yn * offs_yn[:, None] + stride_yo * offs_yo[None, :]
    y_mask = (offs_yn[:, None] < N) & (offs_yo[None, :] < O)
    tl.store(y_ptrs, y, mask=y_mask)

""" Triton Dense Kernel Launchers """
def _triton_dense_launcher_fp32(
    x: torch.Tensor, 
    w: torch.Tensor,
    best_config: triton.Config) -> torch.Tensor:

    assert x.shape[1] == w.shape[0], "Incompatible dimensions"
    assert x.is_contiguous(), "Matrix X must be contiguous"
    assert w.is_contiguous(), "Matrix W must be contiguous"
    assert x.dtype == torch.float32
    assert w.dtype == torch.float32

    N, I = x.shape
    I, O = w.shape
    y = torch.empty((N, O), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(N, best_config.kwargs['BLOCK_SIZE_N']) * triton.cdiv(O, best_config.kwargs['BLOCK_SIZE_O']), )
    _triton_dense_kernel_fp32[grid](
        x, w, y,
        N, O, I,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_SIZE_N=best_config.kwargs['BLOCK_SIZE_N'],
        BLOCK_SIZE_O=best_config.kwargs['BLOCK_SIZE_O'],
        BLOCK_SIZE_I=best_config.kwargs['BLOCK_SIZE_I'],
        GROUP_SIZE_N=best_config.kwargs['GROUP_SIZE_N'],
        num_warps=best_config.num_warps,
        num_stages=best_config.num_stages
    )
    return y

def _triton_dense_launcher_fp16(
    x: torch.Tensor, 
    w: torch.Tensor,
    config: triton.Config = None) -> torch.Tensor:
    
    assert x.shape[1] == w.shape[0], "Incompatible dimensions"
    assert x.is_contiguous(), "Matrix X must be contiguous"
    assert w.is_contiguous(), "Matrix W must be contiguous"
    assert x.dtype == torch.bfloat16
    assert w.dtype == torch.bfloat16

    N, I = x.shape
    I, O = w.shape
    y = torch.empty((N, O), device=x.device, dtype=x.dtype)
    if config is None:
        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']) * triton.cdiv(O, META['BLOCK_SIZE_O']), )
        _triton_dense_kernel_fp16[grid](
            x, w, y,
            N, O, I,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            y.stride(0), y.stride(1)
        )
    else:
        grid = (triton.cdiv(N, config.kwargs['BLOCK_SIZE_N']) * triton.cdiv(O, config.kwargs['BLOCK_SIZE_O']), )
        _triton_dense_kernel_fp16_no_autotune[grid](
            x, w, y,
            N, O, I,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_SIZE_N=config.kwargs['BLOCK_SIZE_N'],
            BLOCK_SIZE_O=config.kwargs['BLOCK_SIZE_O'],
            BLOCK_SIZE_I=config.kwargs['BLOCK_SIZE_I'],
            GROUP_SIZE_N=config.kwargs['GROUP_SIZE_N'],
            num_warps=config.num_warps,
            num_stages=config.num_stages
        )

    return y

def _triton_dense_load_store_launcher_fp32(
    x: torch.Tensor, 
    w: torch.Tensor, 
    best_config: triton.Config) -> torch.Tensor:

    assert x.shape[1] == w.shape[0], "Incompatible dimensions"
    assert x.is_contiguous(), "Matrix X must be contiguous"
    assert w.is_contiguous(), "Matrix W must be contiguous"
    assert x.dtype == torch.float32
    assert w.dtype == torch.float32

    N, I = x.shape
    I, O = w.shape
    y = torch.empty((N, O), device=x.device, dtype=x.dtype)
    _triton_dense_load_store_kernel_fp32[(triton.cdiv(N, best_config.kwargs['BLOCK_SIZE_N']) * triton.cdiv(O, best_config.kwargs['BLOCK_SIZE_O']), )](
        x, w, y,
        N, O, I,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_SIZE_N=best_config.kwargs['BLOCK_SIZE_N'],
        BLOCK_SIZE_O=best_config.kwargs['BLOCK_SIZE_O'],
        BLOCK_SIZE_I=best_config.kwargs['BLOCK_SIZE_I'],
        GROUP_SIZE_N=best_config.kwargs['GROUP_SIZE_N'],
        num_warps=best_config.num_warps,
        num_stages=best_config.num_stages
    )
    return y

def _triton_dense_load_store_launcher_fp16(
    x: torch.Tensor, 
    w: torch.Tensor, 
    best_config: triton.Config) -> torch.Tensor:

    assert x.shape[1] == w.shape[0], "Incompatible dimensions"
    assert x.is_contiguous(), "Matrix X must be contiguous"
    assert w.is_contiguous(), "Matrix W must be contiguous"
    assert x.dtype == torch.bfloat16
    assert w.dtype == torch.bfloat16

    N, I = x.shape
    I, O = w.shape
    y = torch.empty((N, O), device=x.device, dtype=x.dtype)
    _triton_dense_load_store_kernel_fp16[(triton.cdiv(N, best_config.kwargs['BLOCK_SIZE_N']) * triton.cdiv(O, best_config.kwargs['BLOCK_SIZE_O']), )](
        x, w, y,
        N, O, I,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_SIZE_N=best_config.kwargs['BLOCK_SIZE_N'],
        BLOCK_SIZE_O=best_config.kwargs['BLOCK_SIZE_O'],
        BLOCK_SIZE_I=best_config.kwargs['BLOCK_SIZE_I'],
        GROUP_SIZE_N=best_config.kwargs['GROUP_SIZE_N'],
        num_warps=best_config.num_warps,
        num_stages=best_config.num_stages
    )
    return y

""" Triton Dense Functions """
def triton_dense_fp32(
    x: torch.Tensor, 
    w: torch.Tensor) -> torch.Tensor:
    
    best_config = getattr(_triton_dense_kernel_fp16, 'best_config', None)

    if best_config is None:
        default_config = triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_O': 32,  'BLOCK_SIZE_I': 32, 'GROUP_SIZE_N': 8}, num_stages=5, num_warps=2)
        print(f"No autotuned config found. Running with default config: {default_config}")
        print("Ensure to run triton_dense_fp16 before")
        best_config = default_config

    x = x.flatten(0, -2)
    return _triton_dense_launcher_fp32(x, w, best_config)

def triton_dense_fp16(
    x: torch.Tensor, 
    w: torch.Tensor,
    config: triton.Config = None) -> torch.Tensor:

    x = x.flatten(0, -2)
    return _triton_dense_launcher_fp16(x, w, config)

def triton_dense_load_store_fp32(
    x: torch.Tensor, 
    w: torch.Tensor) -> torch.Tensor:

    best_config = getattr(_triton_dense_kernel_fp32, 'best_config', None)

    if best_config is None:
        default_config = triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_O': 32,  'BLOCK_SIZE_I': 32, 'GROUP_SIZE_N': 8}, num_stages=5, num_warps=2)
        print(f"No autotuned config found. Running with default config: {default_config}")
        print("Ensure to run triton_dense_fp32 before")
        best_config = default_config
    
    x = x.flatten(0, -2)
    return _triton_dense_load_store_launcher_fp32(x, w, best_config)

def triton_dense_load_store_fp16(
    x: torch.Tensor, 
    w: torch.Tensor) -> torch.Tensor:

    best_config = getattr(_triton_dense_kernel_fp16, 'best_config', None)

    if best_config is None:
        default_config = triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_O': 32,  'BLOCK_SIZE_I': 32, 'GROUP_SIZE_N': 8}, num_stages=5, num_warps=2)
        print(f"No autotuned config found. Running with default config: {default_config}")
        print("Ensure to run triton_dense_fp16 before")
        best_config = default_config
    
    x = x.flatten(0, -2)
    return _triton_dense_load_store_launcher_fp16(x, w, best_config)

""" Get Triton Dense Kernel Autotuned Configuration """
def get_triton_dense_fp16_config():
    return getattr(_triton_dense_kernel_fp16, 'best_config', None)