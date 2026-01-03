import torch
import triton
import triton.language as tl
from liger_kernel.ops.utils import ensure_contiguous

# 假设 ub_manager.py 在当前目录或 python path 下
# 这里的 import 路径请根据你的实际项目结构调整
from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy

# -----------------------------------------------------------------------------
# Kernels (保持高性能的 1D Flatten 实现)
# -----------------------------------------------------------------------------

@triton.jit
def _swiglu_forward_kernel_flat(
    a_ptr, b_ptr, c_ptr, total_elements, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    
    # Grid-Stride Loop
    start_idx = pid * BLOCK_SIZE
    stride = num_progs * BLOCK_SIZE

    for idx in range(start_idx, total_elements, stride):
        offsets = idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements
        # Ascend 强烈建议使用 FP32 进行计算
        a_val = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b_val = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        res = (a_val * tl.sigmoid(a_val)) * b_val
        tl.store(c_ptr + offsets, res, mask=mask)

@triton.jit
def _swiglu_backward_kernel_flat(
    dc_ptr, a_ptr, b_ptr, da_ptr, db_ptr, total_elements, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    start_idx = pid * BLOCK_SIZE
    stride = num_progs * BLOCK_SIZE
    
    for idx in range(start_idx, total_elements, stride):
        offsets = idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements
        
        dc = tl.load(dc_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        a  = tl.load(a_ptr  + offsets, mask=mask, other=0.0).to(tl.float32)
        b  = tl.load(b_ptr  + offsets, mask=mask, other=0.0).to(tl.float32)
        
        sig_a = tl.sigmoid(a)
        silu_a = a * sig_a
        term1 = silu_a * (1.0 - sig_a) + sig_a
        
        db = dc * silu_a
        da = dc * b * term1
        
        tl.store(da_ptr + offsets, da, mask=mask)
        tl.store(db_ptr + offsets, db, mask=mask)

# -----------------------------------------------------------------------------
# 辅助函数：调用 compute_default_tiling_strategy
# -----------------------------------------------------------------------------

def get_optimal_block_size(total_elements, is_backward=False):
    """
    利用现有的 compute_default_tiling_strategy 计算最佳 Block Size
    """
    # 1. 设定内存倍率 (Memory Multiplier)
    # Forward 比较轻，Backward 需要存更多中间变量，所以给更高的倍率
    # 这里的 8.0 和 12.0 是根据 910B UB (192KB) 推导出的经验值，能稳定算出 4096
    multiplier = 12.0 if is_backward else 8.0
    
    # 2. 调用计算函数
    # 我们把输入看作 1D 的 (total_elements,)，只对第0维 Tiling
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.9,       # 安全边际，留 10% 给系统
        dtype_size=4,            # 强制按 FP32 (4 bytes) 计算，保证安全
        memory_multiplier=multiplier,
        shapes=((total_elements,),),
        tiling_dims=(0,)
    )
    
    # 3. 解析结果
    if tile_shapes and len(tile_shapes) > 0:
        # 结果形如 ((4096,),)
        block_size = tile_shapes[0][0]
        # 限制下限，太小没效率
        return max(256, block_size)
    else:
        # 兜底
        return 2048

def get_npu_core_count():
    try:
        props = triton.runtime.driver.active.utils.get_device_properties(0)
        return props.get("num_vectorcore", 20)
    except:
        return 20

# -----------------------------------------------------------------------------
# Python Wrapper
# -----------------------------------------------------------------------------

def swiglu_forward(a, b):
    if not a.is_contiguous(): a = a.contiguous()
    if not b.is_contiguous(): b = b.contiguous()
    
    total_elements = a.numel()
    c = torch.empty_like(a)
    
    # 【兼容点】直接调用你现有的库函数计算 Block Size
    block_size = get_optimal_block_size(total_elements, is_backward=False)
    
    num_cores = get_npu_core_count()
    grid_size = min(num_cores, (total_elements + block_size - 1) // block_size)

    _swiglu_forward_kernel_flat[(grid_size,)](
        a, b, c,
        total_elements,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=3 
    )
    return c

def swiglu_backward(a, b, dc):
    if not dc.is_contiguous(): dc = dc.contiguous()
    if not a.is_contiguous(): a = a.contiguous()
    if not b.is_contiguous(): b = b.contiguous()

    total_elements = dc.numel()
    grad_a = torch.empty_like(a)
    grad_b = torch.empty_like(b)
    
    # 【兼容点】Backward 倍率更高，Block Size 可能会自动变小（如果在老硬件上）
    block_size = get_optimal_block_size(total_elements, is_backward=True)

    num_cores = get_npu_core_count()
    grid_size = min(num_cores, (total_elements + block_size - 1) // block_size)

    _swiglu_backward_kernel_flat[(grid_size,)](
        dc, a, b,
        grad_a, grad_b,
        total_elements,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=3
    )
    return grad_a, grad_b

# -----------------------------------------------------------------------------
# Autograd Function
# -----------------------------------------------------------------------------

class LigerSiLUMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        c = swiglu_forward(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        grad_a, grad_b = swiglu_backward(a, b, dc)
        return grad_a, grad_b
