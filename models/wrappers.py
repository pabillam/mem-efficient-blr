import torch
import torch.nn as nn
import importlib

class MonarchLinear(nn.Module):
    def __init__(self, in_features, out_features, b, rank, func_path, device, dtype, bias=True, is_compiled=False):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.in_features = in_features
        self.out_features = out_features
        self.b = b
        assert in_features % b == 0 and out_features % b == 0
        assert rank % b == 0

        self.rank = rank
        module_name, func_name = func_path.rsplit(".", 1)

        module = importlib.import_module(module_name)
        self.func_name = func_name
        self.func = getattr(module, func_name)
        self.is_compiled = is_compiled
        if self.is_compiled:
            self.func = torch.compile(self.func)

        if "triton" in self.func_name:
            self.w1_bfly = nn.Parameter(torch.randn((self.b, self.in_features // self.b, self.rank), device=self.device, dtype=self.dtype))
            if "left" in self.func_name:
                self.w2_bfly = nn.Parameter(torch.randn((self.b, self.rank, self.out_features // self.b), device=self.device, dtype=self.dtype))
            else:
                self.w2_bfly = nn.Parameter(torch.randn((self.b, self.out_features // self.b, self.rank), device=self.device, dtype=self.dtype))
        elif "torch" in self.func_name:
            self.w1_bfly = nn.Parameter(torch.randn((self.b, self.rank, self.in_features // self.b), device=self.device, dtype=self.dtype))
            self.w2_bfly = nn.Parameter(torch.randn((self.b, self.out_features // self.b, self.rank), device=self.device, dtype=self.dtype))

        if bias:
            self.bias = nn.Parameter(torch.randn(self.out_features, device=self.device, dtype=self.dtype))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        if "triton" in self.func_name:
            out = self.func(x, self.w1_bfly, self.w2_bfly)
        elif "torch" in self.func_name:
            out, _, _, _ = self.func(x, self.w1_bfly, self.w2_bfly)

        if self.bias is not None:
            out += self.bias.to(x.dtype)
        return out

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, rank={self.rank}, b={self.b}, func={self.func_name}"

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, func_path, device, dtype, bias=True, is_compiled=False):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.in_features = in_features
        self.out_features = out_features

        self.rank = rank
        module_name, func_name = func_path.rsplit(".", 1)

        module = importlib.import_module(module_name)
        self.func_name = func_name
        self.func = getattr(module, func_name)
        self.is_compiled = is_compiled
        if self.is_compiled:
            self.func = torch.compile(self.func)

        if "triton" in self.func_name:
            self.V = nn.Parameter(torch.randn((self.in_features, self.rank), device=self.device, dtype=self.dtype))
            self.U = nn.Parameter(torch.randn((self.rank, self.out_features), device=self.device, dtype=self.dtype))
        elif "torch" in self.func_name:
            self.V = nn.Parameter(torch.randn((self.rank, self.in_features), device=self.device, dtype=self.dtype))
            self.U = nn.Parameter(torch.randn((self.out_features, self.rank), device=self.device, dtype=self.dtype))

        if bias:
            self.bias = nn.Parameter(torch.randn(self.out_features, device=self.device, dtype=self.dtype))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        out = self.func(x, self.V, self.U)

        if self.bias is not None:
            out += self.bias.to(x.dtype)
        return out

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, rank={self.rank}, func={self.func_name}"

class DenseLinear(nn.Module):
    def __init__(self, in_features, out_features, func_path, device, dtype, bias=True, is_compiled=False):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.in_features = in_features
        self.out_features = out_features

        module_name, func_name = func_path.rsplit(".", 1)

        module = importlib.import_module(module_name)
        self.func_name = func_name
        self.func = getattr(module, func_name)
        self.is_compiled = is_compiled
        if self.is_compiled:
            self.func = torch.compile(self.func)

        if "triton" in self.func_name:
            self.W = nn.Parameter(torch.randn((self.in_features, self.out_features), device=self.device, dtype=self.dtype))
        elif "torch" in self.func_name:
            self.W = nn.Parameter(torch.randn((self.out_features, self.in_features), device=self.device, dtype=self.dtype))

        if bias:
            self.bias = nn.Parameter(torch.randn(self.out_features, device=self.device, dtype=self.dtype))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        out = self.func(x, self.W)

        if self.bias is not None:
            out += self.bias.to(x.dtype)
        return out

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, func={self.func_name}"

class BlastLinear(nn.Module):
    def __init__(self, in_features, out_features, b, rank, func_path, device, dtype, bias=True, is_compiled=False):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.in_features = in_features
        self.out_features = out_features
        self.b = b
        assert in_features % b == 0 and out_features % b == 0

        self.rank = rank

        module_name, func_name = func_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        self.func_name = func_name
        self.func = getattr(module, func_name)
        self.is_compiled = is_compiled
        if self.is_compiled:
            self.func = torch.compile(self.func)

        if "triton" in self.func_name:
            if "bmm" in self.func_name:
                self.U = nn.Parameter(torch.randn((self.b, self.out_features // self.b, self.rank), device=self.device, dtype=self.dtype))
                self.V = nn.Parameter(torch.randn((self.b, self.in_features // self.b, self.rank), device=self.device, dtype=self.dtype))
                self.S = nn.Parameter(torch.randn((self.rank, self.b, self.b), device=self.device, dtype=self.dtype))
            else:
                self.U = nn.Parameter(torch.randn((self.b, self.rank, self.out_features // self.b), device=self.device, dtype=self.dtype))
                self.V = nn.Parameter(torch.randn((self.b, self.in_features // self.b, self.rank), device=self.device, dtype=self.dtype))
                self.S = nn.Parameter(torch.randn((self.b, self.b, self.rank), device=self.device, dtype=self.dtype))
        elif "torch" in self.func_name:
            self.U = nn.Parameter(torch.randn((self.b, self.out_features // self.b, self.rank), device=self.device, dtype=self.dtype))
            self.V = nn.Parameter(torch.randn((self.b, self.rank, self.in_features // self.b), device=self.device, dtype=self.dtype))
            self.S = nn.Parameter(torch.randn((self.b, self.b, self.rank), device=self.device, dtype=self.dtype))

        if bias:
            self.bias = nn.Parameter(torch.randn(self.out_features, device=self.device, dtype=self.dtype))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        if "triton" in self.func_name:
            out = self.func(x, self.U, self.V, self.S)
        elif "torch" in self.func_name:
            out, _, _ = self.func(x, self.U, self.V, self.S)

        if self.bias is not None:
            out += self.bias.to(x.dtype)
        return out

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, rank={self.rank}, b={self.b}, func={self.func_name}"
