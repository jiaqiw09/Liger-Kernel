# 初学者模板：`layer_norm.py` 算子实现练习文件（仅包含中文分步指导）

"""
目标：
- 理解 LayerNorm 的前向/后向数学表达
- 为 `layer_norm_forward` 与 `layer_norm_backward` 编写数值与梯度测试

测试要点：
- 前向：与 `torch.nn.LayerNorm` 或手动实现比较输出
- 后向：验证 `layer_norm_backward` 的导数（数值差分或 gradcheck）
- 注意 rstd/mean 的 dtype 与数值稳定性（kernel 中对 fp32 的使用）

使用：
- 补齐 TODO 并运行：`pytest -q src/liger_kernel/ops/layer_norm_test.py`
"""

import torch
import pytest
import torch.nn as nn
import math


def manual_layer_norm(X, W, B, eps):
    # X: (..., hidden)
    mean = X.mean(dim=-1, keepdim=True)
    var = X.var(dim=-1, unbiased=False, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    X_norm = (X - mean) * rstd
    Y = X_norm * W + B
    return Y, mean.squeeze(-1), rstd.squeeze(-1)


def test_forward_matches_pytorch():
    # TODO:
    # 1) 构造小规模输入 X (e.g., shape=(2, 8))，W,B
    # 2) 调用 layer_norm_forward 并与 manual_layer_norm / nn.LayerNorm 比较
    assert True


def test_backward_gradients():
    # TODO:
    # 1) 使用 gradcheck 或者比较 fused backward 与 PyTorch autograd 的梯度
    # 2) 注意 kernel 内部对 fp32 的转换，测试时可使用 float32/float64
    assert True

# 运行：
# pytest -q src/liger_kernel/ops/layer_norm_test.py
