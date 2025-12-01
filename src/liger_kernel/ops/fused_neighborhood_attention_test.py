# 初学者模板：`fused_neighborhood_attention.py` 算子实现练习文件（仅包含中文分步指导）

"""
目标：
- 理解 neighborhood attention 的计算流程（QK、mask、softmax、AV）以及与标准全局 attention 的区别
- 为前向/后向实现写小规模数值对比测试和梯度测试

测试要点：
1) 生成 neighborhood mask：验证 `_neighborhood_mask_kernel` 的行为是否与预期的邻域窗口一致（支持 dilation）
2) 前向数值对比：对比 `fused_neighborhood_attention_forward` 与显式（naive）实现
   - naive 实现步骤：
     a) 计算 Q @ K^T，得到 scores
     b) 用 mask 将不在邻域内的位置设为 -inf
     c) softmax(scores * scale)
     d) output = softmax @ V
3) 后向验证：使用 small-scale 输入做梯度检查或数值差分，比较 fused 的 backward 与显式计算的 grad
4) 边界情况：seq_len 较小或 kernel_size 较大时的行为，dilation=1 vs >1

使用：
- 按 TODO 补齐并运行：`pytest -q src/liger_kernel/ops/fused_neighborhood_attention_test.py`
"""

import torch
import pytest
import math
import torch.nn.functional as F


def naive_neighborhood_attention(query, key, value, kernel_size=7, dilation=1, scale=None):
    # query/key/value shape: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = query.shape
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    out = torch.zeros_like(query)
    attn_weights = torch.zeros(batch_size, num_heads, seq_len, seq_len, device=query.device, dtype=query.dtype)

    half = kernel_size // 2
    for b in range(batch_size):
        for h in range(num_heads):
            for i in range(seq_len):
                # compute neighbor positions with dilation
                neighbors = []
                for offset in range(-half, half + 1):
                    pos = i + offset * dilation
                    if 0 <= pos < seq_len:
                        neighbors.append(pos)
                # compute scores only for neighbors
                q = query[b, h, i : i + 1]  # (1, head_dim)
                k_neigh = key[b, h, neighbors]  # (len(neigh), head_dim)
                scores = (q @ k_neigh.transpose(-2, -1)).squeeze(0) * scale
                probs = F.softmax(scores, dim=-1)
                v_neigh = value[b, h, neighbors]
                out[b, h, i] = probs @ v_neigh
                # fill attn_weights
                attn_weights[b, h, i, neighbors] = probs
    return out, attn_weights


def test_forward_against_naive():
    # TODO: 补充具体实现步骤：
    # 1) 构造小规模输入，例如 batch_size=1, num_heads=1, seq_len=8, head_dim=16
    # 2) 设置 kernel_size=3, dilation=1
    # 3) 调用 fused_neighborhood_attention_forward
    #    from src.liger_kernel.ops.fused_neighborhood_attention import fused_neighborhood_attention_forward
    # 4) 用 naive_neighborhood_attention 计算参考值并比较 output 与 attn_weights
    assert True


def test_neighborhood_mask_generation():
    # TODO: 测试 `_neighborhood_mask_kernel` 生成的 mask 是否与手动生成的 mask 一致
    # - 例如 seq_len=7, kernel_size=3, dilation=2 等组合
    assert True


def test_backward_small_scale():
    # TODO: 使用 gradcheck 或有限差分方法在非常小的输入上验证 backward
    # 注意：fused 函数可能依赖 Triton kernel，不方便直接用 gradcheck；
    # 可以比较在 fused 实现上手动计算的 grad 与 naive 实现使用 autograd 得到的 grad。
    assert True

# 运行：
# pytest -q src/liger_kernel/ops/fused_neighborhood_attention_test.py
