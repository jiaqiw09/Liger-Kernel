# 初学者模板：`rms_norm.py` 算子实现练习文件（仅包含中文分步指导）

"""
目标：
- 理解 RMSNorm 的数学公式：rstd = 1 / sqrt(mean(x^2) + eps)，Y = (x * rstd) * (offset + W)
- 为前向/后向实现编写数值对比测试（包括 casting_mode 的影响）

测试要点：
- 前向：对比 `rms_norm_forward` 与手动实现的结果（Y、RSTD）
- 后向：验证 `rms_norm_backward` 的 dX、dW 在小规模下与有限差分或 autograd 的结果接近
- casting_mode：在不同 casting_mode 下的行为（'llama','gemma','none'）

使用：
- 补齐 TODO 并运行：`pytest -q src/liger_kernel/ops/rms_norm_test.py`
"""

import torch
import pytest


def manual_rms_norm_forward(X, W, eps, offset=0.0):
    # X: (n_rows, n_cols)
    n_rows, n_cols = X.shape
    Y = torch.zeros_like(X)
    RSTD = torch.zeros(n_rows, dtype=torch.float32)
    for i in range(n_rows):
        x = X[i]
        mean_sq = torch.mean(x * x)
        rstd = 1.0 / torch.sqrt(mean_sq + eps)
        RSTD[i] = rstd
        Y[i] = (x * rstd) * (offset + W)
    return Y, RSTD


def test_forward_matches_manual():
    # TODO:
    # 1) 构造 small-scale X (e.g., (2,8)), W shape (8,), eps
    # 2) 调用 rms_norm_forward 并与 manual_rms_norm_forward 比较 Y、RSTD
    assert True


def test_backward_numerical():
    # TODO: 对 dX/dW 使用数值差分评估 backward 有没有正确
    # - 准备 small-scale 示例并对 X 的分量与 W 作微小扰动估算数值梯度
    assert True

# 运行：
# pytest -q src/liger_kernel/ops/rms_norm_test.py
