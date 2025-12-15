import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import ensure_contiguous

# BLOCK_ROWS: Handled per program_id to reduce Grid size (CoreDim limit).
# BLOCK_SIZE_SUB: Tiling size to ensure memory usage fits in Unified Buffer (UB).
BLOCK_ROWS = 128
BLOCK_SIZE_SUB = 1024


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _swiglu_forward_kernel(
    a_ptr, b_ptr, c_ptr, stride, n_rows, n_cols: tl.constexpr, BLOCK_ROWS: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)
    row_start = pid * BLOCK_ROWS

    for r in range(BLOCK_ROWS):
        curr_row = row_start + r
        row_mask = curr_row < n_rows

        row_offset = curr_row * stride

        # Tile columns to prevent UB overflow
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
    BLOCK_ROWS: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    row_start = pid * BLOCK_ROWS

    for r in range(BLOCK_ROWS):
        curr_row = row_start + r
        row_mask = curr_row < n_rows  # Boolean scalar mask for the row

        row_offset = curr_row * stride

        for off in range(0, n_cols, BLOCK_SIZE_SUB):
            col_offsets = off + tl.arange(0, BLOCK_SIZE_SUB)

            mask = (col_offsets < n_cols) & row_mask

            dc_row = tl.load(dc_ptr + row_offset + col_offsets, mask=mask, other=0.0)
            a_row = tl.load(a_ptr + row_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)
            b_row = tl.load(b_ptr + row_offset + col_offsets, mask=mask, other=0.0)

            sig_a = tl.sigmoid(a_row)
            silu_a = a_row * sig_a

            db_row = dc_row * silu_a
            da_row = dc_row * (silu_a * (1.0 - sig_a) + sig_a) * b_row

            tl.store(da_ptr + row_offset + col_offsets, da_row, mask=mask)
            tl.store(db_ptr + row_offset + col_offsets, db_row, mask=mask)


def swiglu_forward(a, b):
    ori_shape = a.shape
    n_cols = ori_shape[-1]

    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    grid = (triton.cdiv(n_rows, BLOCK_ROWS),)

    _swiglu_forward_kernel[grid](
        a,
        b,
        c,
        c.stride(0),
        n_rows,
        n_cols=n_cols,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_SIZE_SUB=BLOCK_SIZE_SUB,
    )
    return a, b, c.view(*ori_shape)


def swiglu_backward(a, b, dc):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]

    dc = dc.view(-1, n_cols)
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    n_rows = dc.shape[0]

    grad_a = torch.empty_like(a)
    grad_b = torch.empty_like(b)

    grid = (triton.cdiv(n_rows, BLOCK_ROWS),)

    _swiglu_backward_kernel[grid](
        dc,
        a,
        b,
        grad_a,
        grad_b,
        dc.stride(0),
        n_rows,
        n_cols=n_cols,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_SIZE_SUB=BLOCK_SIZE_SUB,
    )
    return grad_a.view(*ori_shape), grad_b.view(*ori_shape)


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
