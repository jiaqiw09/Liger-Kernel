# 初学者模板：`dyt.py` 算子实现练习文件（仅包含中文分步指导）

"""
目标：
- 理解 `dyt.py` 中算子的数学定义与 Triton kernel 的接口
- 给出如何实现并测试前向/后向（包括带/不带 `beta` 的情况）的详尽步骤

主要符号（从源码可见）：
- `_dyt_fwd_kernel`, `_dyt_bwd_kernel` (Triton kernel)
- `liger_dyt_fwd`, `liger_dyt_bwd`
- `LigerDyTFunction`

实现提示概览：
1) 数学形式：y = tanh(alpha * x) * gamma + (beta if provided)
2) 前向测试：把 `liger_dyt_fwd` 的输出与基于 PyTorch 的逐元素实现对比
3) 后向测试：用 `torch.autograd.gradcheck` 或有限差分验证 `LigerDyTFunction` 的 backward

使用说明：
- 把下面 TODO 部分补齐后，运行 `pytest -q src/liger_kernel/ops/dyt_test.py`
"""

import torch
import pytest


def naive_dyt_forward(x, alpha, gamma, beta):
    # 逐元素参考实现：y = tanh(alpha * x) * gamma + beta
    # 注意：gamma 和 beta 应支持广播（通常为 shape [V]）
    return torch.tanh(alpha * x) * gamma + (beta if beta is not None else 0.0)


def test_forward_matches_naive():
    # TODO: 补充具体实现步骤：
    # 1) 构造小规模输入，例如 x.shape = (2, 4)，确保 dtype=float32，device 为可用的 CUDA 或 XPU
    # 2) 构造 gamma(shape=[4]), alpha标量张量, beta可为 None 或 shape=[4]
    # 3) 调用被测函数： from src.liger_kernel.ops.dyt import liger_dyt_fwd
    # 4) 计算参考值：naive_dyt_forward
    # 5) 使用相对/绝对误差断言两个结果接近
    assert True  # TODO: 替换为实际断言


def test_backward_gradcheck():
    # TODO: 使用 `torch.autograd.gradcheck` 来校验 autograd 实现
    # 注意：gradcheck 要求 double 精度输入并且 requires_grad=True
    # 步骤：
    # - 准备 double 精度输入，例如 x = torch.randn((1, 3), dtype=torch.double, requires_grad=True)
    # - alpha, gamma, beta 也转换为 double，并根据需要设置 requires_grad=True
    # - 将 forward 包装成能被 gradcheck 调用的 lambda
    # - 运行 gradcheck
    assert True  # TODO: 替换为实际 gradcheck 断言


def test_shape_and_contiguity():
    # TODO: 测试非连续输入的处理：
    # - 给定一个非 contiguous 的输入，检查 `liger_dyt_fwd` 是否能正确处理（源代码中有 `assert x.is_contiguous()`）
    # - 若实现里有 `.contiguous()` 转换，验证输出形状一致
    assert True


# 调试与性能提示：
# - 在实现或调试数值差异时，先在 CPU 或仅使用 PyTorch 逐元素实现进行比对；确认数学表达式一致后再调试 Triton kernel。
# - 对于大型向量 N，Triton 的 BLOCK_N、num_warps、num_stages 等超参会影响性能，调试时不要频繁改变这些参数。
# - 注意 dtype（float16/float32）带来的数值差异，调试时可临时使用 float32 或 float64 进行更高精度校验。

# 参考运行：
# pytest -q src/liger_kernel/ops/dyt_test.py
