import torch
import triton
import triton.language as tl

# 尝试导入 torch_npu
try:
    import torch_npu
except ImportError:
    pass

from liger_kernel.ops.utils import ensure_contiguous

# ==========================================
# 1. Triton Kernels (带 Autotune)
# ==========================================

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

# --- Forward Kernel ---
# 使用进阶 Autotune:
# key: 输入大小变化时重新调优
# split_params: 核间切分参数 (对应 XBLOCK)，由 tl.program_id 决定
# tiling_params: 核内切分参数 (对应 XBLOCK_SUB)，由 range/arange 决定
# low_dims: 低维轴，参与了 arange 计算
@triton.autotune(
    configs=[], # 留空，让 Triton-Ascend 自动生成候选配置
    key={"x": "xnumel"},
    split_params={"x": "XBLOCK"},
    tiling_params={"x": "XBLOCK_SUB"},
    low_dims=["x"],
    persistent_reduction=False,
    dual_reduction=False,
)
@triton.jit
def _swiglu_forward_kernel(
    a_ptr, b_ptr, c_ptr, 
    xnumel, 
    XBLOCK: tl.constexpr,      # Autotune 会自动传入这个值
    XBLOCK_SUB: tl.constexpr   # Autotune 会自动传入这个值
):
    # 切分轴逻辑: 基于 program_id
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    
    # 分块轴逻辑: 基于 range 循环
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        base_idx = xoffset + xoffset_sub 
        # 低维轴逻辑: 基于 arange
        offsets = base_idx + tl.arange(0, XBLOCK_SUB)
        mask = offsets < xnumel
        
        a_row = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b_row = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        c_row = silu(a_row).cast(b_row.dtype) * b_row
        tl.store(c_ptr + offsets, c_row, mask=mask)

# --- Backward Kernel ---
# 同样的 Autotune 配置
@triton.autotune(
    configs=[],
    key={"x": "xnumel"},
    split_params={"x": "XBLOCK"},
    tiling_params={"x": "XBLOCK_SUB"},
    low_dims=["x"],
    persistent_reduction=False,
    dual_reduction=False,
)
@triton.jit
def _swiglu_backward_kernel(
    dc_ptr, a_ptr, b_ptr,
    da_ptr, db_ptr,
    xnumel, 
    XBLOCK: tl.constexpr, 
    XBLOCK_SUB: tl.constexpr
):
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        base_idx = xoffset + xoffset_sub
        offsets = base_idx + tl.arange(0, XBLOCK_SUB)
        mask = offsets < xnumel

        dc_row = tl.load(dc_ptr + offsets, mask=mask, other=0.0)
        a_row = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b_row = tl.load(b_ptr + offsets, mask=mask, other=0.0)

        sig_a = tl.sigmoid(a_row)
        silu_a = a_row * sig_a
        
        db_row = dc_row * silu_a
        term = silu_a * (1.0 - sig_a) + sig_a
        da_row = dc_row * term * b_row

        tl.store(da_ptr + offsets, da_row, mask=mask)
        tl.store(db_ptr + offsets, db_row, mask=mask)

# ==========================================
# 2. Python Wrappers
# ==========================================

def swiglu_forward(a, b):
    ori_shape = a.shape
    xnumel = a.numel()

    a_flat = a.view(-1)
    b_flat = b.view(-1)
    c_flat = torch.empty_like(a_flat)

    # 动态 Grid 计算：
    # 因为 XBLOCK 是由 Autotune 决定的，所以 Grid Size = ceil(Total / XBLOCK)
    # meta['XBLOCK'] 会在运行时被 Autotuner 填充
    grid = lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']), )
    
    # 注意：调用时不传 XBLOCK 和 XBLOCK_SUB，让 Autotuner 自动注入
    _swiglu_forward_kernel[grid](
        a_flat, b_flat, c_flat,
        xnumel=xnumel 
        # XBLOCK=... (自动解析)
        # XBLOCK_SUB=... (自动解析)
    )
    
    return a, b, c_flat.view(*ori_shape)

def swiglu_backward(a, b, dc):
    ori_shape = dc.shape
    xnumel = dc.numel()
    
    a_flat = a.view(-1)
    b_flat = b.view(-1)
    dc_flat = dc.view(-1)
    
    da_flat = torch.empty_like(a_flat)
    db_flat = torch.empty_like(b_flat)

    grid = lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']), )

    _swiglu_backward_kernel[grid](
        dc_flat, a_flat, b_flat,
        da_flat, db_flat,
        xnumel=xnumel
    )
    
    return da_flat.view(*ori_shape), db_flat.view(*ori_shape)

# ==========================================
# 3. Autograd Function
# ==========================================

class LigerSiLUMulFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, a, b):
        a, b, c = swiglu_forward(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        grad_a, grad_b = swiglu_backward(a, b, dc)
        return grad_a, grad_b
