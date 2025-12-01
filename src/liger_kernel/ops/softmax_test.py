# 初学者模板：`softmax.py` 算子实现练习文件（仅包含中文分步指导）

"""
目标：
- 验证 `_softmax_forward` 与 `_softmax_backward` 的数值正确性（与 PyTorch 的 softmax / 自动微分 对比）
- 覆盖 single-block 与 multi-block 两种运行路径（通过改变 n_cols 来触发）

测试要点：
- 前向：`_softmax_forward(x)` 输出应等同于 `torch.softmax(x, dim=-1)` 在数值上接近
- 后向：对随机 dy 与 y，使用 `_softmax_backward` 计算 dx 并与基于 softmax 的手动反向公式比较
  - 手动反向公式：dx = y * (dy - sum(dy * y, dim=-1, keepdim=True))
- 边界：处理 -inf（例如在 attention mask 场景）以及非常小/大的值导致的数值稳定性

使用：
- 补齐 TODO 并运行：`pytest -q src/liger_kernel/ops/softmax_test.py`
"""

import torch
import pytest


def manual_softmax(x):
    # 直接使用 PyTorch 来作为参考实现
    return torch.softmax(x, dim=-1)


def manual_softmax_backward(dy, y):
    dot = (dy * y).sum(dim=-1, keepdim=True)
    return y * (dy - dot)


def test_forward_matches_torch():
    # TODO:
    # 1) 构造多个 n_cols 的输入（例如 n_cols=8(小), n_cols=1024(大)）以触发不同 kernel 路径
    # 2) 调用 _softmax_forward 并与 torch.softmax 比较
    assert True


def test_backward_matches_manual():
    # TODO:
    # 1) 对于给定的 x, 先计算 y=_softmax_forward(x)
    # 2) 随机生成 dy，调用 _softmax_backward(dy, y, BLOCK_SIZE, num_warps, multi_block_launch)
    # 3) 用 manual_softmax_backward 计算参考 dx 并比较
    assert True

# 运行：
# pytest -q src/liger_kernel/ops/softmax_test.py
