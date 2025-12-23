import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver
from liger_kernel.ops.utils import ensure_contiguous

# UB 估算：
# 1 block = BLOCK_SIZE * 4 bytes (fp32)
# 需要 3 个 tensor (a, b, c)
# 开启 num_stages=3
# 总占用 = 4096 * 4 * 3 * 3 ≈ 144 KB < 192KB (安全)
NPU_MAX_BLOCK_SIZE = 4096
SAFE_DEFAULT_CORES = 20

def get_npu_utils():
    try:
        props = driver.active.utils.get_device_properties(0)
        if "num_vectorcore" in props:
            return props["num_vectorcore"]
        if "num_aicore" in props:
            return props["num_aicore"]
        return SAFE_DEFAULT_CORES
    except Exception:
        return SAFE_DEFAULT_CORES

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

# -----------------------------------------------------------------------------
# 优化 V2: 纯净流水线版
# -----------------------------------------------------------------------------
@triton.jit
def _swiglu_forward_kernel_opt(
    a_ptr,
    b_ptr,
    c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    num_progs = tl.num_programs(0)
    
    # 简单的 1D 循环，移除复杂的 parallel(0,2)
    # 让编译器专注于生成高效的 MTE 指令流水
    for start_idx in range(pid * BLOCK_SIZE, n_elements, num_progs * BLOCK_SIZE):
        
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load
        # 配合 num_stages=3，这里会自动预取下一轮的数据
        a_val = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b_val = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        
        # Compute
        res = silu(a_val).cast(b_val.dtype) * b_val
        
        # Store
        tl.store(c_ptr + offsets, res, mask=mask)

@triton.jit
def _swiglu_backward_kernel_opt(
    dc_ptr,
    a_ptr,
    b_ptr,
    da_ptr,
    db_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    num_progs = tl.num_programs(0)
    
    for start_idx in range(pid * BLOCK_SIZE, n_elements, num_progs * BLOCK_SIZE):
        
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        dc_val = tl.load(dc_ptr + offsets, mask=mask, other=0.0)
        a_val = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b_val = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        
        sig_a = tl.sigmoid(a_val)
        silu_a = a_val * sig_a
        
        term1 = silu_a * (1.0 - sig_a) + sig_a
        
        db_val = dc_val * silu_a
        da_val = dc_val * term1 * b_val
        
        tl.store(da_ptr + offsets, da_val, mask=mask)
        tl.store(db_ptr + offsets, db_val, mask=mask)

# -----------------------------------------------------------------------------
# 调用逻辑
# -----------------------------------------------------------------------------

def swiglu_forward(a, b):
    c = torch.empty(a.shape, dtype=a.dtype, device=a.device)
    n_elements = a.numel()
    
    # 使用 4096 作为块大小
    block_size = NPU_MAX_BLOCK_SIZE
    num_cores = get_npu_utils()
    target_grid = min(num_cores, (n_elements + block_size - 1) // block_size)
    
    # 【关键改动】num_stages=3
    # 这会指示 Triton 分配 3 个 buffer 进行轮替预取
    _swiglu_forward_kernel_opt[(target_grid,)](
        a, b, c,
        n_elements,
        BLOCK_SIZE=block_size,
        num_stages=3,  # <--- 核心改动：开启流水线
    )
    return a, b, c

def swiglu_backward(a, b, dc):
    grad_a = torch.empty(a.shape, dtype=a.dtype, device=a.device)
    grad_b = torch.empty(b.shape, dtype=b.dtype, device=b.device)
    
    n_elements = dc.numel()
    block_size = NPU_MAX_BLOCK_SIZE
    num_cores = get_npu_utils()
    target_grid = min(num_cores, (n_elements + block_size - 1) // block_size)

    _swiglu_backward_kernel_opt[(target_grid,)](
        dc, a, b, grad_a, grad_b,
        n_elements,
        BLOCK_SIZE=block_size,
        num_stages=3,  # <--- 核心改动
    )
    return grad_a, grad_b

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
