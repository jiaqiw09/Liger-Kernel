import torch
import triton
import triton.language as tl

# 尝试导入 torch_npu
try:
    import torch_npu
except ImportError:
    pass

from liger_kernel.ops.utils import ensure_contiguous

def get_npu_core_count():
    default_cores = 20 
    if hasattr(torch, 'npu') and torch.npu.is_available():
        try:
            return torch_npu.npu.get_device_properties(torch.npu.current_device()).multi_processor_count
        except:
            return default_cores
    return default_cores

# ==========================================
# 1. Triton Kernels
# ==========================================

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

@triton.jit
def _swiglu_forward_kernel(
    a_ptr, b_ptr, c_ptr, 
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
        
        a_row = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b_row = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        c_row = silu(a_row).cast(b_row.dtype) * b_row
        tl.store(c_ptr + offsets, c_row, mask=mask)

# --- [修正] Backward Kernel 增加输出指针 ---
@triton.jit
def _swiglu_backward_kernel(
    dc_ptr, a_ptr, b_ptr,        # 输入：梯度，原输入a，原输入b
    da_ptr, db_ptr,              # [新增] 输出：a的梯度，b的梯度
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

        # 加载数据
        dc_row = tl.load(dc_ptr + offsets, mask=mask, other=0.0)
        a_row = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b_row = tl.load(b_ptr + offsets, mask=mask, other=0.0)

        # 重计算
        sig_a = tl.sigmoid(a_row)
        silu_a = a_row * sig_a
        
        # 计算梯度
        # db = dc * silu(a)
        db_row = dc_row * silu_a
        
        # da = dc * b * (silu(a) * (1 - sig(a)) + sig(a))
        term = silu_a * (1.0 - sig_a) + sig_a
        da_row = dc_row * term * b_row

        # [修正] 存入专用的梯度内存，而不是覆盖原输入
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

    num_core = get_npu_core_count()
    XBLOCK = (xnumel + num_core - 1) // num_core
    XBLOCK_SUB = 4096 

    grid = (num_core, )
    
    _swiglu_forward_kernel[grid](
        a_flat, b_flat, c_flat,
        xnumel,
        XBLOCK=XBLOCK,
        XBLOCK_SUB=XBLOCK_SUB
    )
    
    return a, b, c_flat.view(*ori_shape)

# --- [修正] Backward Wrapper 分配新内存 ---
def swiglu_backward(a, b, dc):
    ori_shape = dc.shape
    xnumel = dc.numel()
    
    a_flat = a.view(-1)
    b_flat = b.view(-1)
    dc_flat = dc.view(-1)
    
    # [新增] 分配存放梯度的显存
    da_flat = torch.empty_like(a_flat)
    db_flat = torch.empty_like(b_flat)

    num_core = get_npu_core_count()
    XBLOCK = (xnumel + num_core - 1) // num_core
    XBLOCK_SUB = 4096 

    grid = (num_core, )

    # 传入新的指针
    _swiglu_backward_kernel[grid](
        dc_flat, a_flat, b_flat,
        da_flat, db_flat,          # 传入 da, db
        xnumel,
        XBLOCK=XBLOCK,
        XBLOCK_SUB=XBLOCK_SUB
    )
    
    return da_flat.view(*ori_shape), db_flat.view(*ori_shape)

# ==========================================
# 3. Autograd Function
# ==========================================

class LigerSiLUMulFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, a, b):
        # 此时 a, b 也是被 swiglu_forward 用过的，
        # 但 swiglu_forward 只是读取，没有修改，所以 safe
        a, b, c = swiglu_forward(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        # 现在返回的是全新的 tensor，原来的 a, b 保持不变
        grad_a, grad_b = swiglu_backward(a, b, dc)
        return grad_a, grad_b
