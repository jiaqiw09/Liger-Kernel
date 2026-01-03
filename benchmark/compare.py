import torch
import torch_npu
from liger_swiglu_npu import LigerSiLUMulFunction

# 1. 设置环境
device = "npu:0"
torch.npu.set_device(device)

# 2. 准备数据 (Case 3: 4096, 4, 4096, bf16)
shape = (4096, 4, 4096)
dtype = torch.bfloat16

# 输入数据
x_native = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
x_liger = x_native.detach().clone().requires_grad_(True)

# 梯度数据 (SwiGLU输出维度减半)
grad_out = torch.randn((4096, 4, 2048), dtype=dtype, device=device)

print(f"Start Profiling Case: {shape} {dtype}")

# ==========================================
# Part A: Native NPU SwiGLU (原生算子)
# ==========================================
print(">>> Running Native torch_npu.npu_swiglu ...")
torch.npu.synchronize()
for _ in range(20):
    # Forward
    y = torch_npu.npu_swiglu(x_native, dim=-1)
    # Backward
    y.backward(grad_out)
    # 清空梯度 (模拟训练)
    x_native.grad = None
torch.npu.synchronize()

# ==========================================
# Part B: Liger Triton SwiGLU (自定义算子)
# ==========================================
print(">>> Running Liger Triton SwiGLU ...")
torch.npu.synchronize()
for _ in range(20):
    # Forward (Liger Kernel需要手动切分)
    a, b = torch.chunk(x_liger, 2, dim=-1)
    y = LigerSiLUMulFunction.apply(a, b)
    # Backward
    y.backward(grad_out)
    # 清空梯度
    x_liger.grad = None
torch.npu.synchronize()

print("Done.")
