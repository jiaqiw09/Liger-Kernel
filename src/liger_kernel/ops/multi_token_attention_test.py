# 初学者模板：`multi_token_attention.py` 算子实现练习文件（仅包含中文分步指导）

"""
目标：
- 理解 multi-token attention 的流程，包括：
  - 对 scores 做 tri-mask（upper triangle -> -inf 或 0）
  - 用 softmax 或 sparsemax 得到注意力分布
  - 用 conv2d 将注意力映射到输出，再对输出做 mask
- 为前向与后向实现编写小规模对比测试

测试要点：
1) `_mask_inf_forward` / `_mask_inf_backward`：与手动上三角设 -inf 的实现对比
2) `_mask_zero_forward` / `_mask_zero_backward`：与手动上三角设 0 的实现对比
3) 整体 forward：对比使用 softmax 的路径与手动按步骤计算（scores->mask->softmax->conv2d->mask）
4) backward：在 small-scale 下验证 grad_scores 的值（softmax 路径）或 sparsemax 路径

使用：
- 补齐 TODO 并运行：`pytest -q src/liger_kernel/ops/multi_token_attention_test.py`
"""

import torch
import pytest
import torch.nn.functional as F
from torch.nn import Conv2d


def manual_mask_inf(scores):
    # 将上三角（col > row）设为 -1e9
    out = scores.clone()
    *batch, L, _ = out.shape
    for idx in range(out.view(-1, L, L).shape[0]):
        mat = out.view(-1, L, L)[idx]
        for i in range(L):
            for j in range(L):
                if j > i:
                    mat[i, j] = -1e9
    return out


def manual_mask_zero(scores):
    out = scores.clone()
    *batch, L, _ = out.shape
    for idx in range(out.view(-1, L, L).shape[0]):
        mat = out.view(-1, L, L)[idx]
        for i in range(L):
            for j in range(L):
                if j > i:
                    mat[i, j] = 0.0
    return out


def test_mask_forward_backward_equivalence():
    # TODO:
    # 1) 构造 small-scale scores，如 shape=(1, 6, 6)
    # 2) 比较 _mask_inf_forward 与 manual_mask_inf
    # 3) 比较 _mask_zero_forward 与 manual_mask_zero
    # 4) 对比 _mask_inf_backward/_mask_zero_backward 与手动反向（例如上三角置 0）
    assert True


def test_full_forward_softmax_path():
    # TODO:
    # 1) 构造 scores，weight（conv2d 权重），bias
    # 2) 手动按步骤执行：scores -> mask_inf -> softmax -> conv2d -> mask_zero
    # 3) 调用 LigerMultiTokenAttentionFunction.forward(compare outputs)
    assert True

# 运行：
# pytest -q src/liger_kernel/ops/multi_token_attention_test.py
