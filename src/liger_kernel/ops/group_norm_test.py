# 初学者模板：`group_norm.py` 算子实现练习文件（仅包含中文分步指导）

"""
目标：
- 理解 GroupNorm 的数学公式与实现细节
- 编写前向与后向的数值与梯度测试（与 PyTorch 的 `nn.GroupNorm` 或手动实现对比）

测试要点：
- 前向：对比 `group_norm_forward` 与 `torch.nn.GroupNorm` 的输出（注意 shape 转换）
- 后向：验证 `group_norm_backward` 或 `LigerGroupNormFunction` 的 gradients 与 PyTorch autograd 的结果一致
- 边界：channels % num_groups == 0 的断言、不同 dtype（float32/float16）的行为

使用：
- 补齐 TODO 并运行：`pytest -q src/liger_kernel/ops/group_norm_test.py`
"""

import torch
import pytest
import torch.nn as nn


def manual_group_norm(X, num_channels, num_groups, W, B, eps):
    # 手动实现 GroupNorm 的前向以便对比（用于小规模测试）
    batch_size, channels, hidden = X.shape
    assert channels == num_channels
    channels_per_group = num_channels // num_groups
    X_view = X.view(batch_size, num_groups, -1)
    mean = X_view.mean(dim=-1, keepdim=True)
    var = X_view.var(dim=-1, unbiased=False, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    X_norm = (X_view - mean) * rstd
    X_norm = X_norm.view(batch_size, channels, hidden)
    # W/B shape: (num_channels,)
    W_broadcast = W.view(1, num_channels, 1)
    B_broadcast = B.view(1, num_channels, 1)
    Y = X_norm * W_broadcast + B_broadcast
    return Y, mean.squeeze(-1), rstd.squeeze(-1)


def test_forward_matches_manual_and_pytorch():
    # TODO:
    # 1) 构造 small-scale 输入：batch_size=2, num_channels=4, hidden=6, num_groups=2
    # 2) 创建 W, B
    # 3) 调用 group_norm_forward 并与 manual_group_norm 及 torch.nn.GroupNorm 输出比较
    assert True


def test_backward_gradients():
    # TODO:
    # 1) 使用 torch.autograd 在 PyTorch 的 GroupNorm 层上计算 gradients
    # 2) 使用 LigerGroupNormFunction 的 forward，然后调用 backward 或使用 gradcheck 来比较
    assert True

# 运行：
# pytest -q src/liger_kernel/ops/group_norm_test.py
