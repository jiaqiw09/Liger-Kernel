# 初学者模板：`geglu.py` 算子实现练习文件（仅包含中文分步指导）

"""
目标：
- 理解 GEGlu（Gated GELU）算子的具体数值表达：c = GELU(a) * b，且这里使用 tanh 近似的 GELU 形式
- 为前向与后向实现撰写详细测试（与手动实现的公式对比、梯度校验）

使用方法：
- 按照 TODO 补齐测试并运行：`pytest -q src/liger_kernel/ops/geglu_test.py`
"""

import torch
import pytest
import math


def tanh_gelu_approx(a):
    # 源码使用的 tanh 近似 GELU：0.5 * a * (1 + tanh(sqrt(2/pi)*(a + 0.044715*a^3)))
    sqrt_2_over_pi = 0.7978845608028654
    return 0.5 * a * (1 + torch.tanh(sqrt_2_over_pi * (a + 0.044715 * a * a * a)))


def naive_geglu(a, b):
    return tanh_gelu_approx(a) * b


def test_forward_matches_naive():
    # TODO:
    # 1) 构造随机小规模输入 a, b（例如 shape=(2, 8)），dtype=float32
    # 2) 调用 geglu_forward（from src.liger_kernel.ops.geglu import geglu_forward）
    # 3) 计算 naive_geglu，并断言输出接近
    assert True


def test_backward_gradcheck():
    # TODO:
    # 1) 使用 torch.autograd.gradcheck 检查 LigerGELUMulFunction 的正反向一致性
    # 2) gradcheck 要求 double 精度并设置 requires_grad=True
    # 3) 或者使用数值差分来验证 geglu_backward 的输出
    assert True

# 提示：
# - 源码在 kernel 中对 a 进行了 cast 到 float32，最后再 cast 回 b 的 dtype（若不同）。在测试时注意 dtype 转换可能带来的微小误差。
# - 若在 float16 上观察到显著误差，先在 float32/float64 上调试验证实现正确性。

# 运行：
# pytest -q src/liger_kernel/ops/geglu_test.py
