"""
High-Performance UB-Aware SwiGLU for Ascend NPU (Standalone).

Optimization Strategy:
1. Grid Size = Physical Core Count (Vector Cores).
   - Prevents task scheduling overhead.
   - Uses Grid-Stride Loop to process data continuously.
2. Flattened 1D Memory Access.
   - Ignores rows/cols, treats memory as a single contiguous block.
   - Maximizes memory bandwidth (coalesced access).
3. Fixed UB-Safe Block Size + Pipeline.
   - Uses fixed block size (4096) to fit safely within ~192KB UB.
   - Uses num_stages=3 to hide HBM latency via prefetching.
"""

import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver

# -----------------------------------------------------------------------------
# Configuration Constants
# -----------------------------------------------------------------------------

# Unified Buffer (UB) Limits on Ascend:
# Typically ~192KB on 910B.
# FP32 (4 bytes) * 4096 elements * 8 tensors (worst case backward) ~= 128KB.
# This leaves room for system overhead and double buffering.
UB_SAFE_BLOCK_SIZE = 4096

# Fallback core count if detection fails
SAFE_DEFAULT_CORES = 20

# -----------------------------------------------------------------------------
# Hardware Detection Helper
# -----------------------------------------------------------------------------

def get_npu_core_count():
    """
    Detects the number of Vector Cores on the current Ascend device.
    """
    try:
        # Get properties for device 0 (current)
        props = driver.active.utils.get_device_properties(0)
        
        # Priority 1: Vector Cores (Primary compute unit for SwiGLU)
        if "num_vectorcore" in props:
            return props["num_vectorcore"]
        
        # Priority 2: AI Cores (On some older firmwares)
        if "num_aicore" in props:
            return props["num_aicore"]
            
        return SAFE_DEFAULT_CORES
    except Exception:
        # If Triton driver check fails, try torch_npu (if available in newer versions)
        return SAFE_DEFAULT_CORES

# -----------------------------------------------------------------------------
# Triton Kernels (Flattened 1D + Grid-Stride)
# -----------------------------------------------------------------------------

@triton.jit
def _swiglu_forward_kernel_flat(
    a_ptr, 
    b_ptr, 
    c_ptr, 
    total_elements, 
    BLOCK_SIZE: tl.constexpr
):
    # 1. Thread Identity
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    
    # 2. Grid-Stride Logic
    # Each core processes a chunk, then skips ahead by (GridSize * BlockSize)
    # This acts like a persistent thread consuming the work queue.
    start_idx = pid * BLOCK_SIZE
    stride = num_progs * BLOCK_SIZE

    for idx in range(start_idx, total_elements, stride):
        # Offsets for the current block
        offsets = idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        # Load & Cast to FP32 (Crucial for Ascend performance/precision)
        a_val = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b_val = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # Compute: SwiGLU = (a * sigmoid(a)) * b
        res = (a_val * tl.sigmoid(a_val)) * b_val

        # Store
        tl.store(c_ptr + offsets, res, mask=mask)


@triton.jit
def _swiglu_backward_kernel_flat(
    dc_ptr, 
    a_ptr, 
    b_ptr, 
    da_ptr, 
    db_ptr,
    total_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    
    start_idx = pid * BLOCK_SIZE
    stride = num_progs * BLOCK_SIZE

    for idx in range(start_idx, total_elements, stride):
        offsets = idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        # Load inputs (FP32)
        dc = tl.load(dc_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        a  = tl.load(a_ptr  + offsets, mask=mask, other=0.0).to(tl.float32)
        b  = tl.load(b_ptr  + offsets, mask=mask, other=0.0).to(tl.float32)

        # Recompute Forward Activation
        sig_a = tl.sigmoid(a)
        silu_a = a * sig_a
        
        # Gradient Math
        # d(SiLU)/da = SiLU + Sigmoid * (1 - SiLU/a * a) -> SiLU + Sig * (1 - Sig * a)
        # Simplified term: silu * (1 - sig) + sig
        term1 = silu_a * (1.0 - sig_a) + sig_a
        
        # db = dc * SiLU(a)
        db_res = dc * silu_a
        # da = dc * b * term1
        da_res = dc * b * term1

        # Store Gradients
        tl.store(da_ptr + offsets, da_res, mask=mask)
        tl.store(db_ptr + offsets, db_res, mask=mask)

# -----------------------------------------------------------------------------
# Python Launchers
# -----------------------------------------------------------------------------

def swiglu_forward(a, b):
    # Ensure memory is contiguous for 1D flattening
    if not a.is_contiguous(): a = a.contiguous()
    if not b.is_contiguous(): b = b.contiguous()
    
    total_elements = a.numel()
    c = torch.empty_like(a)

    # Strategy: Use all physical cores
    num_cores = get_npu_core_count()
    
    # Calculate Grid Size
    # If data is small, don't launch more cores than blocks
    needed_blocks = (total_elements + UB_SAFE_BLOCK_SIZE - 1) // UB_SAFE_BLOCK_SIZE
    grid_size = min(num_cores, needed_blocks)

    _swiglu_forward_kernel_flat[(grid_size,)](
        a, b, c,
        total_elements,
        BLOCK_SIZE=UB_SAFE_BLOCK_SIZE,
        num_warps=4,   # 4 Warps is standard for vector ops
        num_stages=3   # Enable Pipeline (Ping-Pong buffering) to hide latency
    )
    return c

def swiglu_backward(a, b, dc):
    if not dc.is_contiguous(): dc = dc.contiguous()
    if not a.is_contiguous(): a = a.contiguous()
    if not b.is_contiguous(): b = b.contiguous()

    total_elements = dc.numel()
    grad_a = torch.empty_like(a)
    grad_b = torch.empty_like(b)
    
    num_cores = get_npu_core_count()
    needed_blocks = (total_elements + UB_SAFE_BLOCK_SIZE - 1) // UB_SAFE_BLOCK_SIZE
    grid_size = min(num_cores, needed_blocks)

    _swiglu_backward_kernel_flat[(grid_size,)](
        dc, a, b,
        grad_a, grad_b,
        total_elements,
        BLOCK_SIZE=UB_SAFE_BLOCK_SIZE,
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

def liger_swiglu(a, b):
    return LigerSiLUMulFunction.apply(a, b)

# -----------------------------------------------------------------------------
# Benchmark & Verification Main Block
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    
    # 1. Setup
    torch.manual_seed(123)
    device = "npu:0"
    dtype = torch.float16
    
    # Shape Config: Llama 3 70B (SwiGLU specific)
    # Batch * SeqLen = 32 * 4096 tokens
    # Hidden Dim = 14336 (Intermediate size for 70B model)
    B, S, H = 32, 4096, 14336 
    print(f"--- Benchmark Configuration ---")
    print(f"Device: {torch.npu.get_device_name(0)}")
    print(f"Shape: [{B*S}, {H}] (Flattened)")
    print(f"Dtype: {dtype}")
    print(f"Cores detected: {get_npu_core_count()}")
    print("-" * 30)

    # Create Tensors
    # We use 'view' to simulate the input shape of an MLP layer
    a = torch.randn((B * S, H), device=device, dtype=dtype, requires_grad=True)
    b = torch.randn((B * S, H), device=device, dtype=dtype, requires_grad=True)
    
    # -------------------------------------------------------------------------
    # 2. Correctness Check
    # -------------------------------------------------------------------------
    print("Running Correctness Check...")
    
    # Baseline (Torch Native)
    import torch.nn.functional as F
    def torch_swiglu_fn(x, y):
        return F.silu(x) * y

    target_c = torch_swiglu_fn(a, b)
    loss_native = target_c.sum()
    loss_native.backward()
    grad_a_native = a.grad.clone()
    grad_b_native = b.grad.clone()
    
    # Reset Grads
    a.grad = None
    b.grad = None
    
    # Triton (Liger)
    triton_c = liger_swiglu(a, b)
    loss_triton = triton_c.sum()
    loss_triton.backward()
    grad_a_triton = a.grad.clone()
    grad_b_triton = b.grad.clone()

    # Compare
    fwd_diff = (target_c - triton_c).abs().max().item()
    bwd_a_diff = (grad_a_native - grad_a_triton).abs().max().item()
    bwd_b_diff = (grad_b_native - grad_b_triton).abs().max().item()
    
    if fwd_diff < 1e-2 and bwd_a_diff < 1e-2:
        print(f"✅ Correctness Passed! (Max Diff: {fwd_diff:.6f})")
    else:
        print(f"❌ Correctness Failed. Forward Diff: {fwd_diff}, Grad Diff: {bwd_a_diff}")
        exit()

    # -------------------------------------------------------------------------
    # 3. Performance Benchmark
    # -------------------------------------------------------------------------
    print("\nRunning Performance Benchmark (Warmup: 10, Iter: 100)...")

    # Warmup
    for _ in range(10):
        torch_swiglu_fn(a, b)
        liger_swiglu(a, b)
    torch.npu.synchronize()

    # Torch Timing
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(100):
        _ = torch_swiglu_fn(a, b)
    end_event.record()
    torch.npu.synchronize()
    torch_time = start_event.elapsed_time(end_event) / 100.0

    # Triton Timing
    start_event.record()
    for _ in range(100):
        _ = liger_swiglu(a, b)
    end_event.record()
    torch.npu.synchronize()
    triton_time = start_event.elapsed_time(end_event) / 100.0

    print(f"Torch Native: {torch_time:.3f} ms")
    print(f"Triton Liger: {triton_time:.3f} ms")
    print(f"Speedup:      {torch_time / triton_time:.2f}x")