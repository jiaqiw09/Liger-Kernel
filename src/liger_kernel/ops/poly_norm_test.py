# 初学者模板：`poly_norm.py` 算子实现练习文件（仅包含中文分步指导）

"""
目标：
- 理解 PolyNorm 的定义与实现细节：y = w0*norm(x^3) + w1*norm(x^2) + w2*norm(x) + b
- 为前向/后向编写对比测试

测试要点：
- 前向：与手动实现比对输出和缓存的 RSTD（每行 3 个 rstd）
- 后向：对比 poly_norm_backward 与有限差分或 PyTorch autograd（手动实现小规模的解析梯度或者数值差分）

使用：
- 补齐 TODO 并运行：`pytest -q src/liger_kernel/ops/poly_norm_test.py`
"""

import torch
import pytest


def manual_polynorm_forward(X, W, B, eps=1e-6):
    # X: (n_rows, n_cols)
    # W: (3,), B: scalar
    n_rows, n_cols = X.shape
    Y = torch.zeros_like(X)
    RSTD = torch.zeros((n_rows, 3), dtype=torch.float32)
    for i in range(n_rows):
        x = X[i]
        x1 = x
        x2 = x * x
        x3 = x * x * x
        mean3 = torch.mean(x3 * x3)
        rstd3 = 1.0 / torch.sqrt(mean3 + eps)
        mean2 = torch.mean(x2 * x2)
        rstd2 = 1.0 / torch.sqrt(mean2 + eps)
        mean1 = torch.mean(x1 * x1)
        rstd1 = 1.0 / torch.sqrt(mean1 + eps)
        RSTD[i, 0] = rstd3
        RSTD[i, 1] = rstd2
        RSTD[i, 2] = rstd1
        y = W[0] * (x3 * rstd3) + W[1] * (x2 * rstd2) + W[2] * (x1 * rstd1) + B
        Y[i] = y
    return Y, RSTD


def test_forward_matches_manual():
    # TODO:
    # 1) 构造 small-scale X, W (shape (3,)), B
    # 2) 调用 poly_norm_forward 并与 manual_polynorm_forward 比较 Y 和 RSTD
    assert True


def test_backward_numerical():
    # TODO: 对于小规模输入，使用数值差分验证 poly_norm_backward 的 dX, dW, dB
    # - 对 X 的每个分量加上一个很小的 eps, 计算 loss 差值并估算数值梯度
    assert True

# 运行：
# pytest -q src/liger_kernel/ops/poly_norm_test.py
