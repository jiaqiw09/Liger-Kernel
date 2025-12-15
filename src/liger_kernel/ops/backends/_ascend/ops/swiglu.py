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

        if "num_vectorcore" in props:
            return props["num_vectorcore"]

        if "num_aicore" in props:
            return props["num_aicore"]

        return SAFE_DEFAULT_CORES
    except Exception:
        return SAFE_DEFAULT_CORES


def calculate_settings(n_rows, n_cols):
    """
    Calculate optimal Grid and Block settings:
    1. Grid Size = Physical Vector Core count (Maximize parallelism, minimize queuing).
    2. Block Size = Maximize tiling size within UB limits.
    """
    target_block = triton.next_power_of_2(n_cols)
    block_size_sub = min(target_block, NPU_MAX_BLOCK_SIZE)
    block_size_sub = max(block_size_sub, 32)

    target_grid = get_npu_utils()

    grid_size = min(target_grid, n_rows)

    block_rows_per_core = (n_rows + grid_size - 1) // grid_size

    return grid_size, block_rows_per_core, block_size_sub


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
    BLOCK_ROWS_PER_CORE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    row_start = pid * BLOCK_ROWS_PER_CORE

    # Iterate over assigned rows.
    for r in range(BLOCK_ROWS_PER_CORE):
        curr_row = row_start + r

        row_mask = curr_row < n_rows
        row_offset = curr_row * stride

        for off in range(0, n_cols, BLOCK_SIZE_SUB):
            col_offsets = off + tl.arange(0, BLOCK_SIZE_SUB)

            mask = (col_offsets < n_cols) & row_mask

            a_row = tl.load(a_ptr + row_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)
            b_row = tl.load(b_ptr + row_offset + col_offsets, mask=mask, other=0.0)

            c_row = silu(a_row).cast(b_row.dtype) * b_row

            tl.store(c_ptr + row_offset + col_offsets, c_row, mask=mask)


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
    BLOCK_ROWS_PER_CORE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    row_start = pid * BLOCK_ROWS_PER_CORE

    for r in range(BLOCK_ROWS_PER_CORE):
        curr_row = row_start + r

        # Row validation mask
        row_mask = curr_row < n_rows
        row_offset = curr_row * stride

        for off in range(0, n_cols, BLOCK_SIZE_SUB):
            col_offsets = off + tl.arange(0, BLOCK_SIZE_SUB)

            mask = (col_offsets < n_cols) & row_mask

            dc_row = tl.load(dc_ptr + row_offset + col_offsets, mask=mask, other=0.0)
            a_row = tl.load(a_ptr + row_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)
            b_row = tl.load(b_ptr + row_offset + col_offsets, mask=mask, other=0.0)

            # Recompute Forward
            sig_a = tl.sigmoid(a_row)
            silu_a = a_row * sig_a

            term1 = silu_a * (1.0 - sig_a) + sig_a
            db_row = dc_row * silu_a
            da_row = dc_row * term1 * b_row

            tl.store(da_ptr + row_offset + col_offsets, da_row, mask=mask)
            tl.store(db_ptr + row_offset + col_offsets, db_row, mask=mask)


def swiglu_forward(a, b):
    # Use torch.empty() to create a standard contiguous tensor.
    # Avoiding empty_like() prevents inheriting NPU-internal formats (e.g., NZ).
    c = torch.empty(a.shape, dtype=a.dtype, device=a.device)

    # Flatten input to 2D (N, Hidden) to properly handle 3D tensors (Batch, Seq, Hidden)
    n_cols = a.shape[-1]
    a_flat = a.view(-1, n_cols)
    b_flat = b.view(-1, n_cols)
    c_flat = c.view(-1, n_cols)
    n_rows = a_flat.shape[0]

    grid_size, block_rows, block_size_sub = calculate_settings(n_rows, n_cols)

    _swiglu_forward_kernel[(grid_size,)](
        a_flat,
        b_flat,
        c_flat,
        c_flat.stride(0),
        n_rows,
        n_cols,
        BLOCK_ROWS_PER_CORE=block_rows,
        BLOCK_SIZE_SUB=block_size_sub,
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

    grid_size, block_rows, block_size_sub = calculate_settings(n_rows, n_cols)

    _swiglu_backward_kernel[(grid_size,)](
        dc_flat,
        a_flat,
        b_flat,
        grad_a_flat,
        grad_b_flat,
        dc_flat.stride(0),
        n_rows,
        n_cols,
        BLOCK_ROWS_PER_CORE=block_rows,
        BLOCK_SIZE_SUB=block_size_sub,
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
