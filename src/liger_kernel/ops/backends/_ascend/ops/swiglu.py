"""
UB-aware SwiGLU implementation for Ascend NPU.

This implementation automatically adjusts block sizes to fit within UB constraints,
preventing UB overflow errors while maintaining high performance via pipelining.
"""

import torch
import triton
import triton.language as tl

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import ensure_contiguous

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

@triton.jit
def _swiglu_forward_kernel_npu(
    a_ptr, 
    b_ptr, 
    c_ptr, 
    stride, 
    n_cols: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr
):
    """
    UB-aware SwiGLU forward kernel for NPU.
    """
    program_id = tl.program_id(0).to(tl.int64)

    # Locate start index for the current row
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    # Process in tiles to ensure we fit in UB.
    # Because num_stages > 1, the compiler will prefetch future blocks
    # into the UB automatically.
    for i in range(0, n_cols, BLOCK_SIZE):
        col_offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        # Load input (FP16/BF16 -> FP32 for compute accuracy and stability on NPU)
        a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

        # Compute SwiGLU: (a * sigmoid(a)) * b
        res = silu(a_row) * b_row

        # Store result
        tl.store(c_ptr + col_offsets, res, mask=mask)


@triton.jit
def _swiglu_backward_kernel_npu_explicit(
    dc_ptr, 
    a_ptr, 
    b_ptr, 
    da_ptr, # Output buffer for grad_a
    db_ptr, # Output buffer for grad_b
    stride, 
    n_cols: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr
):
    """
    UB-aware SwiGLU backward kernel for NPU with explicit output buffers.
    """
    program_id = tl.program_id(0).to(tl.int64)

    row_offset = program_id * stride
    
    for i in range(0, n_cols, BLOCK_SIZE):
        col_offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Calculate pointers
        curr_dc = dc_ptr + row_offset + col_offsets
        curr_a  = a_ptr  + row_offset + col_offsets
        curr_b  = b_ptr  + row_offset + col_offsets
        curr_da = da_ptr + row_offset + col_offsets
        curr_db = db_ptr + row_offset + col_offsets

        # Load inputs and cast to FP32 for precision
        dc_val = tl.load(curr_dc, mask=mask, other=0.0).to(tl.float32)
        a_val  = tl.load(curr_a, mask=mask, other=0.0).to(tl.float32)
        b_val  = tl.load(curr_b, mask=mask, other=0.0).to(tl.float32)

        # Recomputation of forward activations
        sig_a = tl.sigmoid(a_val)
        silu_a = a_val * sig_a
        
        # SwiGLU Gradient Math:
        # term1 = d(SiLU)/da = silu + sigmoid * (1 - silu/a * a)
        # Simplified to: silu_a * (1.0 - sig_a) + sig_a
        term1 = silu_a * (1.0 - sig_a) + sig_a
        
        # db = dc * SiLU(a)
        db_val = dc_val * silu_a
        # da = dc * b * d(SiLU)/da
        da_val = dc_val * term1 * b_val

        # Store gradients
        tl.store(curr_da, da_val, mask=mask)
        tl.store(curr_db, db_val, mask=mask)


def swiglu_forward(a, b):
    ori_shape = a.shape
    n_cols = ori_shape[-1]
    
    # Flatten logic
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    desired_block_size, num_warps = calculate_settings(n_cols)

    # ========================================================
    # Performance Optimization Strategy:
    # 1. Enable pipelining (NUM_STAGES = 3) to hide memory latency.
    # 2. To prevent UB Overflow, we must scale the memory multiplier.
    #    Since 3 stages mean 3 tiles reside in UB simultaneously,
    #    the memory pressure is 3x per element.
    # ========================================================
    NUM_STAGES = 3
    
    # Base multiplier for a single tile (FP32 conversion + intermediates)
    # 2 inputs + 1 output + FP32 overhead ~= 8x input size (conservative)
    BASE_MEMORY_MULTIPLIER = 8.0 
    
    # Effective multiplier forces the tiling algorithm to pick a smaller BLOCK_SIZE
    # that accommodates 3 stages.
    EFFECTIVE_MULTIPLIER = BASE_MEMORY_MULTIPLIER * NUM_STAGES 

    dtype_size = a.element_size() # e.g., 2 bytes for BF16
    
    shapes = ((n_cols,),)
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.85, 
        dtype_size=dtype_size,
        memory_multiplier=EFFECTIVE_MULTIPLIER, 
        shapes=shapes,
        tiling_dims=(0,),
    )

    if tile_shapes is not None and len(tile_shapes) > 0:
        adjusted_block_size = tile_shapes[0][0]
    else:
        # Fallback: manually reduce block size to fit stages
        adjusted_block_size = max(32, desired_block_size // NUM_STAGES)

    _swiglu_forward_kernel_npu[(n_rows,)](
        a,
        b,
        c,
        c.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=adjusted_block_size,
        num_warps=num_warps,
        num_stages=NUM_STAGES, 
    )
    return a, b, c.view(*ori_shape)


def swiglu_backward(a, b, dc):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    n_rows = dc.shape[0]

    desired_block_size, num_warps = calculate_settings(n_cols)

    # ========================================================
    # Backward Strategy:
    # Requires significantly more memory (3 loads, 2 stores, complex FP32 math).
    # We increase the base multiplier to 14.0.
    # ========================================================
    NUM_STAGES = 3
    BASE_MEMORY_MULTIPLIER = 14.0 
    EFFECTIVE_MULTIPLIER = BASE_MEMORY_MULTIPLIER * NUM_STAGES

    dtype_size = dc.element_size()
    
    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.85,
        dtype_size=dtype_size,
        memory_multiplier=EFFECTIVE_MULTIPLIER,
        shapes=((n_cols,),),
        tiling_dims=(0,),
    )
    
    if tile_shapes:
        adjusted_block_size = tile_shapes[0][0]
    else:
        # Fallback safety
        adjusted_block_size = max(32, desired_block_size // NUM_STAGES)

    # Allocate gradients
    grad_a = torch.empty_like(a) 
    grad_b = torch.empty_like(b)
    
    # Flatten views
    a_flat = a.view(-1, n_cols)
    b_flat = b.view(-1, n_cols)
    ga_flat = grad_a.view(-1, n_cols)
    gb_flat = grad_b.view(-1, n_cols)

    _swiglu_backward_kernel_npu_explicit[(n_rows,)](
        dc,
        a_flat,
        b_flat,
        ga_flat,
        gb_flat,
        dc.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=adjusted_block_size,
        num_warps=num_warps,
        num_stages=NUM_STAGES,
    )
    
    return grad_a.view(*ori_shape), grad_b.view(*ori_shape)


class LigerSiLUMulFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, a, b):
        # Forward pass: compute and save tensors for backward
        a, b, c = swiglu_forward(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        # Backward pass: compute gradients
        a, b = ctx.saved_tensors
        grad_a, grad_b = swiglu_backward(a, b, dc)
        return grad_a, grad_b