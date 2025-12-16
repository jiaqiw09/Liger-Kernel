import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver

from liger_kernel.ops.utils import ensure_contiguous

# UB Safe Threshold:
# 4096 elements * 4 bytes (FP32) * 3 tensors * 2 stages ~= 96KB.
# This fits safely within the 192KB Unified Buffer (UB) limit.
NPU_MAX_BLOCK_SIZE = 4096

# Safe Fallback Core Count:
# Used if API detection fails. 20 is a safe baseline for most Ascend chips.
SAFE_DEFAULT_CORES = 20


def get_npu_utils():
    """
    Retrieve NPU physical properties.
    Prioritizes 'num_vectorcore' since SwiGLU is a vector-heavy operation.
    """
    try:
        props = driver.active.utils.get_device_properties(0)
        # 有些固件版本可能 key 不一样，多做几个 fallback
        if "num_vectorcore" in props:
            return props["num_vectorcore"]
        if "num_aicore" in props:
            return props["num_aicore"]
        return SAFE_DEFAULT_CORES
    except Exception:
        return SAFE_DEFAULT_CORES


def calculate_settings(n_rows, n_cols):
    """
    Calculate optimal Grid and Block settings.
    We no longer need 'block_rows_per_core' because we use Grid-Stride Loop.
    """
    # 1. Calculate Block Size (Tiling on Column dimension)
    target_block = triton.next_power_of_2(n_cols)
    block_size_sub = min(target_block, NPU_MAX_BLOCK_SIZE)
    
    # 【重要】为了配合 tl.parallel(0, 2)，我们需要保证切分后的子块至少为 32 (32*2=64)
    # 否则切分太小会导致 Bank 冲突或效率下降
    block_size_sub = max(block_size_sub, 64)

    # 2. Calculate Grid Size (Number of Cores)
    target_grid = get_npu_utils()
    # 如果行数少于核心数，就只启动行数对应的核，避免空转
    grid_size = min(target_grid, n_rows)

    return grid_size, block_size_sub


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _swiglu_forward_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    stride,
    n_rows,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. 获取当前 Core ID
    pid = tl.program_id(0).to(tl.int64)
    # 2. 获取总 Core 数量
    num_progs = tl.num_programs(0)
    
    # 3. 预计算并行切分的大小
    HALF_BLOCK: tl.constexpr = BLOCK_SIZE // 2

    # 4. Grid-Stride Loop: 每个 Core 处理 row_idx, row_idx + num_progs, ...
    for row_idx in range(pid, n_rows, num_progs):
        
        row_offset = row_idx * stride

        # Tiling Loop: 处理每一行的所有列
        for off in range(0, n_cols, BLOCK_SIZE):
            
            # 【核心优化】开启双发射并行，绑定到两个 Vector Pipe
            for s in tl.parallel(0, 2, bind_sub_block=True):
                # 计算子块的偏移量
                # s=0: 处理前一半; s=1: 处理后一半
                current_off = off + s * HALF_BLOCK + tl.arange(0, HALF_BLOCK)
                
                mask = current_off < n_cols
                
                # Load (注意这里不需要 row_mask，因为外层循环保证了 row_idx < n_rows)
                a_row = tl.load(a_ptr + row_offset + current_off, mask=mask, other=0.0).to(tl.float32)
                b_row = tl.load(b_ptr + row_offset + current_off, mask=mask, other=0.0)

                # Vector Compute
                c_row = silu(a_row).cast(b_row.dtype) * b_row

                # Store
                tl.store(c_ptr + row_offset + current_off, c_row, mask=mask)


@triton.jit
def _swiglu_backward_kernel(
    dc_ptr,
    a_ptr,
    b_ptr,
    da_ptr,
    db_ptr,
    stride,
    n_rows,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    num_progs = tl.num_programs(0)
    
    HALF_BLOCK: tl.constexpr = BLOCK_SIZE // 2

    # Grid-Stride Loop
    for row_idx in range(pid, n_rows, num_progs):
        
        row_offset = row_idx * stride

        for off in range(0, n_cols, BLOCK_SIZE):
            
            # 【核心优化】双路并行
            for s in tl.parallel(0, 2, bind_sub_block=True):
                current_off = off + s * HALF_BLOCK + tl.arange(0, HALF_BLOCK)
                
                mask = current_off < n_cols

                # Load
                dc_row = tl.load(dc_ptr + row_offset + current_off, mask=mask, other=0.0)
                a_row = tl.load(a_ptr + row_offset + current_off, mask=mask, other=0.0).to(tl.float32)
                b_row = tl.load(b_ptr + row_offset + current_off, mask=mask, other=0.0)

                # Recompute Forward & Calculate Gradient
                # 这一段计算量比 Forward 大，并行化收益更明显
                sig_a = tl.sigmoid(a_row)
                silu_a = a_row * sig_a

                # Gradient Logic
                # d(SiLU * B) / dA = B * (SiLU' * dA) 
                # SiLU'(x) = SiLU(x) + sigmoid(x)(1 - SiLU(x)/x) ... 简化为下式
                term1 = silu_a * (1.0 - sig_a) + sig_a
                
                db_row = dc_row * silu_a
                da_row = dc_row * term1 * b_row

                # Store
                tl.store(da_ptr + row_offset + current_off, da_row, mask=mask)
                tl.store(db_ptr + row_offset + current_off, db_row, mask=mask)


def swiglu_forward(a, b):
    # Use torch.empty() to create a standard contiguous tensor.
    c = torch.empty(a.shape, dtype=a.dtype, device=a.device)

    # Flatten input to 2D (N, Hidden)
    n_cols = a.shape[-1]
    a_flat = a.view(-1, n_cols)
    b_flat = b.view(-1, n_cols)
    c_flat = c.view(-1, n_cols)
    n_rows = a_flat.shape[0]

    # 不再计算 block_rows_per_core，改用 Grid-Stride 逻辑
    grid_size, block_size = calculate_settings(n_rows, n_cols)

    _swiglu_forward_kernel[(grid_size,)](
        a_flat,
        b_flat,
        c_flat,
        c_flat.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=block_size,
    )
    return a, b, c.view(*a.shape)


def swiglu_backward(a, b, dc):
    grad_a = torch.empty(a.shape, dtype=a.dtype, device=a.device)
    grad_b = torch.empty(b.shape, dtype=b.dtype, device=b.device)

    n_cols = dc.shape[-1]

    # Flatten to 2D
    dc_flat = dc.view(-1, n_cols)
    a_flat = a.view(-1, n_cols)
    b_flat = b.view(-1, n_cols)
    grad_a_flat = grad_a.view(-1, n_cols)
    grad_b_flat = grad_b.view(-1, n_cols)

    n_rows = dc_flat.shape[0]

    grid_size, block_size = calculate_settings(n_rows, n_cols)

    _swiglu_backward_kernel[(grid_size,)](
        dc_flat,
        a_flat,
        b_flat,
        grad_a_flat,
        grad_b_flat,
        dc_flat.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=block_size,
    )
    return grad_a.view(*a.shape), grad_b.view(*b.shape)


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
